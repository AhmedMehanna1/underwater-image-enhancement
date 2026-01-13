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
import math


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
        self.loss_unsup = nn.L1Loss()
        self.loss_str = MyLoss().cuda()
        self.loss_grad = nn.L1Loss().cuda()
        self.loss_cr = ContrastLoss().cuda()
        self.loss_chroma = ChromaLoss().cuda()
        self.loss_saturation = SaturationLoss().cuda()
        self.loss_gray_world = GrayWorldLoss().cuda()
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        self.get_grad = GetGradientNopadding().cuda()
        self.iqa_metric = pyiqa.create_metric('musiq', as_loss=False).cuda()
        self.iqa_metric.eval()
        for p in self.iqa_metric.parameters():
            p.requires_grad_(False)
        self.musiq = pyiqa.create_metric("musiq", as_loss=False).cuda()
        self.musiq.eval()
        for p in self.musiq.parameters():
            p.requires_grad_(False)
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        # set optimizer and learning rate
        self.optimizer_s = AdamP(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        # self.lr_scheduler_s = lr_scheduler.StepLR(self.optimizer_s, step_size=100, gamma=0.1)
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[100, 150], gamma=0.1)

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996):
        # exponential moving average(EMA)
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

    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name, score_r):
        N = teacher_predict.shape[0]
        grad_state = torch.is_grad_enabled()
        with torch.no_grad():
            score_t = self.iqa_metric(teacher_predict).detach().cpu().numpy()
            score_s = self.iqa_metric(student_predict).detach().cpu().numpy()
        torch.set_grad_enabled(grad_state)
        positive_sample = positive_list.clone()
        for idx in range(0, N):
            if score_t[idx] > score_s[idx]:
                if score_t[idx] > score_r[idx]:
                    positive_sample[idx] = teacher_predict[idx]
                    # update the reliable bank
                    temp_c = np.transpose(teacher_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                    temp_c = np.clip(temp_c, 0, 1)
                    arr_c = (temp_c * 255).astype(np.uint8)
                    arr_c = Image.fromarray(arr_c)
                    arr_c.save('%s' % p_name[idx])
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

            # Save checkpoint

            state = {'arch': type(self.model).__name__,
                     'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optimizer_dict': self.optimizer_s.state_dict(),
                     'teacher_state_dict': self.tmodel.state_dict(),
                     'scheduler_dict': self.lr_scheduler_s.state_dict(),
                     'curiter': self.curiter,
                     'best_psnr': best_psnr,
                     'best_val_loss': best_val_loss,
                     }
            ckpt_name = os.path.join(self.args.save_path, 'ckpt_last.pth')
            torch.save(state, ckpt_name)

            if epoch % self.save_period == 0:
                torch.save(state, os.path.join(self.args.save_path, f'ckpt_epoch_{epoch}.pth'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model (val_psnr={best_psnr:.4f}) val_loss={val_loss:.4f} ...")
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'model_best_student.pth'))
                torch.save(self.tmodel.state_dict(), os.path.join(self.args.save_path, 'model_best_teacher.pth'))

    def _train_epoch(self, epoch):
        total_loss = AverageMeter()
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        total_loss_meter = AverageMeter()
        self.model.train()
        self.freeze_teachers_parameters()

        sup_len = len(self.supervised_loader)
        unsup_len = len(self.unsupervised_loader)
        steps = self.iter_per_epoch

        sup_iter = iter(self.supervised_loader)
        unsup_iter = iter(self.unsupervised_loader)

        if sup_len < steps:
            sup_iter = cycle(self.supervised_loader)
        if unsup_len < steps:
            unsup_iter = cycle(self.unsupervised_loader)

        train_loader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
        # tbar = range(len(self.unsupervised_loader))
        # tbar = tqdm(tbar, ncols=130, leave=True)
        tbar = tqdm(range(steps), ncols=170, leave=True)

        tbar.set_description(
            'Train-Student Epoch {} | Ls: {:.4f} Lu: {:.4f}|'
            .format(epoch, sup_loss.avg, unsup_loss.avg)
        )

        for _ in tbar:
            (img_data, label, img_la) = next(sup_iter)
            (unpaired_data_w, unpaired_data_s, unpaired_la, p_list, p_name) = next(unsup_iter)
            # (img_data, label, img_la), (unpaired_data_w, unpaired_data_s, unpaired_la, p_list, p_name) = next(train_loader)
            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            img_la = Variable(img_la).cuda(non_blocking=True)
            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)
            unpaired_data_w = Variable(unpaired_data_w).cuda(non_blocking=True)
            unpaired_la = Variable(unpaired_la).cuda(non_blocking=True)
            p_list = Variable(p_list).cuda(non_blocking=True)
            # teacher output
            predict_target_u = self.predict_with_out_grad(unpaired_data_w, unpaired_la)
            origin_predict = predict_target_u.detach().clone()
            # student output
            outputs_l, outputs_g = self.model(img_data, img_la)
            outputs_ul, _ = self.model(unpaired_data_s, unpaired_la)
            structure_loss = self.loss_str(outputs_l, label)
            perpetual_loss = self.loss_per(outputs_l, label)
            gradient_loss = self.loss_grad(self.get_grad(outputs_l), self.get_grad(label)) + self.loss_grad(outputs_g,
                                                                                                            self.get_grad(
                                                                                                                label))
            outputs_l_01 = outputs_l.clamp(0, 1)
            label_01 = label.clamp(0, 1)

            chroma_loss = self.loss_chroma(outputs_l_01, label_01)
            saturation_loss = self.loss_saturation(outputs_l_01, label_01)
            gray_world_loss = self.loss_gray_world(outputs_l_01)

            loss_sup = structure_loss \
                       + 0.1 * perpetual_loss \
                       + 0.1 * gradient_loss \
                       + 0.15 * chroma_loss \
                       + 0.05 * saturation_loss \
                       + 0.02 * gray_world_loss
            sup_loss.update(loss_sup.mean().item())
            grad_state = torch.is_grad_enabled()
            with torch.no_grad():
                score_r = self.iqa_metric(p_list).detach().cpu().numpy()
                p_sample = self.get_reliable(predict_target_u, outputs_ul, p_list, p_name, score_r)
            torch.set_grad_enabled(grad_state)
            loss_unsu = self.loss_unsup(outputs_ul, p_sample) + self.loss_cr(outputs_ul, p_sample, unpaired_data_s)
            unsup_loss.update(loss_unsu.mean().item())
            consistency_weight = self.get_current_consistency_weight(epoch)
            total_loss = consistency_weight * loss_unsu + loss_sup
            total_loss = total_loss.mean()
            total_loss_meter.update(total_loss.item())

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description(
                'Train-Student Epoch {} | Ls: {:.4f} Lu: {:.4f}|'
                .format(epoch, sup_loss.avg, unsup_loss.avg)
            )

            del img_data, label, unpaired_data_w, unpaired_data_s, img_la, unpaired_la,
            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter)
                self.curiter = self.curiter + 1

        self.writer.add_scalar('Train_loss', total_loss, global_step=epoch)
        self.writer.add_scalar('sup_loss', sup_loss.avg, global_step=epoch)
        self.writer.add_scalar('unsup_loss', unsup_loss.avg, global_step=epoch)
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

        tbar = tqdm(self.val_loader, ncols=170)
        with torch.no_grad():
            for val_data, val_label, val_la in tbar:
                val_data = val_data.cuda(non_blocking=True)
                val_label = val_label.cuda(non_blocking=True)
                val_la = val_la.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(enabled=getattr(self, "use_amp", False)):
                    val_output, val_g = self.model(val_data, val_la)

                    structure_loss = self.loss_str(val_output, val_label)
                    perpetual_loss = self.loss_per(val_output, val_label)
                    gradient_loss = self.loss_grad(self.get_grad(val_output), self.get_grad(val_label)) + \
                                    self.loss_grad(val_g, self.get_grad(val_label))

                    out01 = val_output.clamp(0, 1)
                    gt01 = val_label.clamp(0, 1)
                    chroma_loss = self.loss_chroma(out01, gt01)
                    saturation_loss = self.loss_saturation(out01, gt01)
                    gray_world_loss = self.loss_gray_world(out01)

                    val_sup_loss = structure_loss \
                                   + 0.10 * perpetual_loss \
                                   + 0.10 * gradient_loss \
                                   + 0.15 * chroma_loss \
                                   + 0.05 * saturation_loss \
                                   + 0.02 * gray_world_loss

                val_sup_loss_meter.update(val_sup_loss.mean().item(), n=val_data.size(0))

                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)

                for recovered in val_output:
                    out_rgb = tensor_to_uint8_rgb(recovered)
                    uiqm_meter.update(uiqm(out_rgb))
                    uciqe_meter.update(uciqe(out_rgb))
                    with torch.no_grad():
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
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
