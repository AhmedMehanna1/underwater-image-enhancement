import os

import torch
import torch.fx
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
import torchvision
import torch.distributed as dist
from torch.optim import lr_scheduler
import PIL.Image as Image
from utils import *
from torch.autograd import Variable
from adamp import AdamP
from torchvision.models import vgg16
from loss.losses import *
from model import GetGradientNopadding
from loss.contrast import ContrastLoss
import pyiqa


# -----------------------------
# Extra texture + color helpers
# -----------------------------
def _rgb_to_ycbcr(x: torch.Tensor):
    """
    x: (B,3,H,W) in [0,1]
    returns y, cb, cr
    """
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


class LaplacianLoss(nn.Module):
    """
    Texture/detail loss: L1 between Laplacian pyramids (single-scale Laplacian).
    Strong at preserving fine edges + micro-textures without over-sharpening.
    """
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("k", k.view(1, 1, 3, 3))

    def forward(self, pred, gt):
        # pred/gt: (B,3,H,W) in [0,1]
        # apply per-channel depthwise conv via groups=3
        k = self.k.repeat(3, 1, 1, 1)
        lp = F.conv2d(pred, k, padding=1, groups=3)
        lg = F.conv2d(gt,   k, padding=1, groups=3)
        return F.l1_loss(lp, lg)


class YLumaLoss(nn.Module):
    """
    Keep luminance (Y) close -> helps contrast/brightness structure without color fighting.
    """
    def forward(self, pred, gt):
        y_p, _, _ = _rgb_to_ycbcr(pred)
        y_g, _, _ = _rgb_to_ycbcr(gt)
        return F.l1_loss(y_p, y_g)


class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch, writer):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 20

        # losses
        self.loss_unsup = nn.L1Loss()
        self.loss_str = MyLoss().cuda()
        self.loss_grad = nn.L1Loss().cuda()
        self.loss_cr = ContrastLoss().cuda()

        # color losses (already in your code)
        self.loss_chroma = ChromaLoss().cuda()
        self.loss_saturation = SaturationLoss().cuda()
        self.loss_gray_world = GrayWorldLoss().cuda()

        # NEW: texture + luma (for better texture + stable brightness)
        self.loss_lap = LaplacianLoss().cuda()
        self.loss_y = YLumaLoss().cuda()

        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.get_grad = GetGradientNopadding().cuda()

        # NR IQA (for reliable bank)
        self.musiq = pyiqa.create_metric("musiq", as_loss=False).cuda()
        self.musiq.eval()
        for p in self.musiq.parameters():
            p.requires_grad_(False)

        # perceptual (texture-ish)
        vgg_model = vgg16(pretrained=True).features[:16].cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()

        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()

        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)

        # optimizer / scheduler
        self.optimizer_s = AdamP(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[100, 150], gamma=0.1)

        # weights (tuned to preserve texture + keep good color)
        # - Laplacian: encourages crisp details
        # - Y-luma: stabilizes brightness/contrast
        # - Color terms are ramped (avoid early “over-color/over-white”)
        self.w_per = 0.12
        self.w_grad = 0.12
        self.w_lap = 0.10
        self.w_y = 0.08

        self.w_chroma_max = 0.15
        self.w_sat_max = 0.12
        self.w_gw_max = 0.08

        # ramp lengths (in epochs)
        self.color_ramp = 50.0  # ramp chroma/sat/gw up smoothly
        self.gw_delay = 15.0    # start GW later (prevents early gray/white collapse)

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image, image_l):
        with torch.no_grad():
            predict_target_ul, _ = self.tmodel(image, image_l)
        return predict_target_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def _ramp(self, epoch, ramp_len):
        if ramp_len <= 0:
            return 1.0
        e = float(np.clip(epoch, 0.0, ramp_len))
        return e / ramp_len

    def _color_weights(self, epoch):
        r = self._ramp(epoch, self.color_ramp)
        w_chroma = self.w_chroma_max * r
        w_sat = self.w_sat_max * r

        # delay GW then ramp
        if epoch <= self.gw_delay:
            w_gw = 0.0
        else:
            r2 = self._ramp(epoch - self.gw_delay, max(1.0, self.color_ramp - self.gw_delay))
            w_gw = self.w_gw_max * r2
        return w_chroma, w_sat, w_gw

    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name, score_r):
        N = teacher_predict.shape[0]
        grad_state = torch.is_grad_enabled()
        with torch.no_grad():
            score_t = self.musiq(teacher_predict).detach().cpu().numpy()
            score_s = self.musiq(student_predict).detach().cpu().numpy()
        torch.set_grad_enabled(grad_state)

        positive_sample = positive_list.clone()
        for idx in range(0, N):
            if score_t[idx] > score_s[idx] and score_t[idx] > score_r[idx]:
                positive_sample[idx] = teacher_predict[idx]
                # update reliable bank
                temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                temp_c = np.clip(temp_c, 0, 1)
                arr_c = (temp_c * 255).astype(np.uint8)
                Image.fromarray(arr_c).save('%s' % p_name[idx])

        del N, score_r, score_s, score_t, teacher_predict, student_predict, positive_list
        return positive_sample

    def train(self):
        best_psnr = -1e9
        best_val_loss = float("inf")
        self.freeze_teachers_parameters()

        if not self.args.resume:
            initialize_weights(self.model)
        else:
            ckpt_path = os.path.join(self.args.save_path, "ckpt_last.pth")
            print("Loading checkpoint: {} ...".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer_s.load_state_dict(checkpoint['optimizer_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.tmodel.load_state_dict(checkpoint['teacher_state_dict'])
            self.lr_scheduler_s.load_state_dict(checkpoint['scheduler_dict'])
            self.curiter = checkpoint.get('curiter', 0)
            best_psnr = checkpoint.get('best_psnr', -1e9)
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))

        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_ave = self._train_epoch(epoch)
            loss_val = float(loss_ave)
            psnr_val, val_loss = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)

            print('[%d] main_loss: %.6f, val psnr: %.6f, val loss: %.6f, lr: %.8f' % (
                epoch, loss_val, val_psnr, val_loss, self.lr_scheduler_s.get_last_lr()[0]))

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f"{name}", param, 0)

            state = {
                'arch': type(self.model).__name__,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_dict': self.optimizer_s.state_dict(),
                'teacher_state_dict': self.tmodel.state_dict(),
                'scheduler_dict': self.lr_scheduler_s.state_dict(),
                'curiter': self.curiter,
                'best_psnr': best_psnr,
                'best_val_loss': best_val_loss,
            }
            torch.save(state, os.path.join(self.args.save_path, 'ckpt_last.pth'))

            if epoch % self.save_period == 0:
                torch.save(state, os.path.join(self.args.save_path, f'ckpt_epoch_{epoch}.pth'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_psnr = val_psnr
                print(f"Saving best model (val_psnr={val_psnr:.4f}) val_loss={val_loss:.4f} ...")
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'model_best_student.pth'))
                torch.save(self.tmodel.state_dict(), os.path.join(self.args.save_path, 'model_best_teacher.pth'))

    def _train_epoch(self, epoch):
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        total_loss_meter = AverageMeter()

        self.model.train()
        self.freeze_teachers_parameters()

        steps = self.iter_per_epoch
        sup_iter = iter(self.supervised_loader)
        unsup_iter = iter(self.unsupervised_loader)

        if len(self.supervised_loader) < steps:
            sup_iter = cycle(self.supervised_loader)
        if len(self.unsupervised_loader) < steps:
            unsup_iter = cycle(self.unsupervised_loader)

        tbar = tqdm(range(steps), ncols=170, leave=True)
        tbar.set_description('Train-Student Epoch {} | Ls: {:.4f} Lu: {:.4f}|'.format(epoch, sup_loss.avg, unsup_loss.avg))

        # ramped color weights (better color without killing textures early)
        w_chroma, w_sat, w_gw = self._color_weights(epoch)

        for _ in tbar:
            (img_data, label, img_la) = next(sup_iter)
            (unpaired_data_w, unpaired_data_s, unpaired_la, p_list, p_name) = next(unsup_iter)

            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            img_la = Variable(img_la).cuda(non_blocking=True)

            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)
            unpaired_data_w = Variable(unpaired_data_w).cuda(non_blocking=True)
            unpaired_la = Variable(unpaired_la).cuda(non_blocking=True)

            p_list = Variable(p_list).cuda(non_blocking=True)

            # teacher output
            predict_target_u = self.predict_with_out_grad(unpaired_data_w, unpaired_la)

            # student output
            outputs_l, outputs_g = self.model(img_data, img_la)
            outputs_ul, _ = self.model(unpaired_data_s, unpaired_la)

            # base losses
            structure_loss = self.loss_str(outputs_l, label)
            perceptual_loss = self.loss_per(outputs_l, label)

            # gradient/edge supervision (already present)
            gradient_loss = self.loss_grad(self.get_grad(outputs_l), self.get_grad(label)) + \
                            self.loss_grad(outputs_g, self.get_grad(label))

            # clamp for stable color losses
            out01 = outputs_l.clamp(0, 1)
            gt01 = label.clamp(0, 1)

            # color losses
            chroma_loss = self.loss_chroma(out01, gt01)
            saturation_loss = self.loss_saturation(out01, gt01)
            gray_world_loss = self.loss_gray_world(out01)

            # NEW texture + luminance
            lap_loss = self.loss_lap(out01, gt01)
            y_loss = self.loss_y(out01, gt01)

            # supervised loss: texture-focused + controlled color
            loss_sup = (
                    structure_loss
                    + self.w_per * perceptual_loss
                    + self.w_grad * gradient_loss
                    + self.w_lap * lap_loss
                    + self.w_y * y_loss
                    + w_sat * saturation_loss
                    + w_chroma * chroma_loss
                    + w_gw * gray_world_loss
            )

            sup_loss.update(loss_sup.mean().item())

            # reliable bank selection
            grad_state = torch.is_grad_enabled()
            with torch.no_grad():
                score_r = self.musiq(p_list).detach().cpu().numpy()
                p_sample = self.get_reliable(predict_target_u, outputs_ul, p_list, p_name, score_r)
            torch.set_grad_enabled(grad_state)

            # unsupervised loss (keep as you have it)
            loss_unsu = self.loss_unsup(outputs_ul, p_sample) + self.loss_cr(outputs_ul, p_sample, unpaired_data_s)
            unsup_loss.update(loss_unsu.mean().item())

            consistency_weight = self.get_current_consistency_weight(epoch)
            total_loss = (consistency_weight * loss_unsu + loss_sup).mean()
            total_loss_meter.update(total_loss.item())

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description('Train-Student Epoch {} | Ls: {:.4f} Lu: {:.4f}|'.format(epoch, sup_loss.avg, unsup_loss.avg))

            # IMPORTANT: fix the dangling comma bug in your original code (it creates a syntax/tuple issue)
            del img_data, label, unpaired_data_w, unpaired_data_s, img_la, unpaired_la, p_list

            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter += 1

        if self.writer is not None:
            self.writer.add_scalar('Train_loss', total_loss_meter.avg, global_step=epoch)
            self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
            self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)
            self.writer.add_scalar('w_chroma', float(w_chroma), global_step=epoch)
            self.writer.add_scalar('w_sat', float(w_sat), global_step=epoch)
            self.writer.add_scalar('w_gw', float(w_gw), global_step=epoch)

        self.lr_scheduler_s.step(epoch=epoch - 1)
        return total_loss_meter.avg

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.tmodel.eval()

        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        uiqm_meter = AverageMeter()
        uciqe_meter = AverageMeter()
        musiq_meter = AverageMeter()
        val_sup_loss_meter = AverageMeter()

        # use the same ramped weights for reporting val loss
        w_chroma, w_sat, w_gw = self._color_weights(epoch)

        tbar = tqdm(self.val_loader, ncols=170)
        with torch.no_grad():
            for val_data, val_label, val_la in tbar:
                val_data = val_data.cuda(non_blocking=True)
                val_label = val_label.cuda(non_blocking=True)
                val_la = val_la.cuda(non_blocking=True)

                val_output, val_g = self.model(val_data, val_la)

                structure_loss = self.loss_str(val_output, val_label)
                perceptual_loss = self.loss_per(val_output, val_label)
                gradient_loss = self.loss_grad(self.get_grad(val_output), self.get_grad(val_label)) + \
                                self.loss_grad(val_g, self.get_grad(val_label))

                out01 = val_output.clamp(0, 1)
                gt01 = val_label.clamp(0, 1)

                chroma_loss = self.loss_chroma(out01, gt01)
                saturation_loss = self.loss_saturation(out01, gt01)
                gray_world_loss = self.loss_gray_world(out01)

                lap_loss = self.loss_lap(out01, gt01)
                y_loss = self.loss_y(out01, gt01)

                val_sup_loss = (
                        structure_loss
                        + self.w_per * perceptual_loss
                        + self.w_grad * gradient_loss
                        + self.w_lap * lap_loss
                        + self.w_y * y_loss
                        + w_sat * saturation_loss
                        + w_chroma * chroma_loss
                        + w_gw * gray_world_loss
                )

                val_sup_loss_meter.update(val_sup_loss.mean().item(), n=val_data.size(0))

                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)

                for recovered in val_output:
                    out_rgb = tensor_to_uint8_rgb(recovered)
                    uiqm_meter.update(uiqm(out_rgb))
                    uciqe_meter.update(uciqe(out_rgb))
                    musiq_v = float(self.musiq(recovered).mean().item())
                    musiq_meter.update(musiq_v)

                psnr_val.extend(to_psnr(val_output, val_label))

                tbar.set_description(
                    f"Eval Epoch {epoch} | ValSupLoss: {val_sup_loss_meter.avg:.4f} | "
                    f"PSNR: {psnr_meter.avg:.4f} SSIM: {ssim_meter.avg:.4f} | "
                    f"UIQM: {uiqm_meter.avg:.4f} UCIQE: {uciqe_meter.avg:.4f} | MUSIQ: {musiq_meter.avg:.4f}"
                )

        if self.writer is not None:
            self.writer.add_scalar('Val_sup_loss', val_sup_loss_meter.avg, global_step=epoch)
            self.writer.add_scalar('Val_psnr', psnr_meter.avg, global_step=epoch)
            self.writer.add_scalar('Val_ssim', ssim_meter.avg, global_step=epoch)
            self.writer.add_scalar('Val_uiqm', uiqm_meter.avg, global_step=epoch)
            self.writer.add_scalar('Val_uciqe', uciqe_meter.avg, global_step=epoch)
            self.writer.add_scalar('Val_musiq', musiq_meter.avg, global_step=epoch)

        return psnr_val, val_sup_loss_meter.avg

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
