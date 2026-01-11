import os
import torch
import torch.fx
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

import cv2
import pyiqa

from dataset_all import make_dataset
# my import
from model import AIMnet
from utils import AverageMeter, compute_psnr_ssim, to_psnr, tensor_to_uint8_rgb, uiqm, uciqe

# -----------------------------
# Dataset (paired)
# -----------------------------
class TestLabeledData(data.Dataset):
    def __init__(self, dataroot, fineSize):
        super().__init__()
        self.root = dataroot
        self.fineSize = fineSize

        self.dir_A = os.path.join(self.root, 'input')
        self.dir_B = os.path.join(self.root, 'GT')
        self.dir_C = os.path.join(self.root, 'LA')

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.C_paths = sorted(make_dataset(self.dir_C))

        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("RGB")
        B = Image.open(self.B_paths[index]).convert("RGB")
        C = Image.open(self.C_paths[index]).convert("RGB")

        resized_a = A.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_b = B.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)
        resized_c = C.resize((self.fineSize, self.fineSize), Image.ANTIALIAS)

        tensor_a = self.transform(resized_a)
        tensor_b = self.transform(resized_b)
        tensor_c = self.transform(resized_c)

        return tensor_a, tensor_b, tensor_c

    def __len__(self):
        return len(self.A_paths)


# -----------------------------
# Main test
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

bz = 1
model_root = 'model/ckpt/model_best_student.pth'
input_root = 'test/euvp-paired/dark'
save_path = 'test/euvp-paired/dark/enhanced'
os.makedirs(save_path, exist_ok=True)

Mydata_ = TestLabeledData(input_root, 256)
data_load = data.DataLoader(Mydata_, batch_size=bz, shuffle=False)

test_psnr = AverageMeter()
test_ssim = AverageMeter()
psnr_val = []

# extra metrics running avg
uiqm_avg_out = AverageMeter()
uciqe_avg_out = AverageMeter()
musiq_avg_out = AverageMeter()

uiqm_avg_in = AverageMeter()
uciqe_avg_in = AverageMeter()
musiq_avg_in = AverageMeter()

model = AIMnet().cuda()
model = nn.DataParallel(model, device_ids=[0, 1])
model.load_state_dict(torch.load(model_root))
model.eval()

# MUSIQ (no grad, eval)
musiq = pyiqa.create_metric("musiq", as_loss=False).cuda()
musiq.eval()
for p in musiq.parameters():
    p.requires_grad_(False)

print('START!')
print('Load model successfully!')

tbar = tqdm(range(len(data_load)), ncols=170, leave=True)
data_load = iter(data_load)

with torch.no_grad():
    for data_idx in tbar:
        data_input, data_label, data_la = next(data_load)
        name = os.path.basename(Mydata_.A_paths[data_idx])

        data_input = Variable(data_input).cuda()
        data_label = Variable(data_label).cuda()
        data_la = Variable(data_la).cuda()

        result, _ = model(data_input, data_la)

        # PSNR/SSIM (paired, output vs GT)
        temp_psnr, temp_ssim, N = compute_psnr_ssim(result, data_label)
        test_psnr.update(temp_psnr, N)
        test_ssim.update(temp_ssim, N)
        psnr_val.extend(to_psnr(result, data_label))

        # UIQM/UCIQE on output (no-reference, CPU)
        out_rgb = tensor_to_uint8_rgb(result)
        uiqm_out = uiqm(out_rgb)
        uciqe_out = uciqe(out_rgb)

        # Also compute same metrics on input (optional but useful)
        in_rgb = tensor_to_uint8_rgb(data_input)
        uiqm_in = uiqm(in_rgb)
        uciqe_in = uciqe(in_rgb)

        # MUSIQ (GPU) on input/output
        musiq_out = float(musiq(result).mean().item())
        musiq_in = float(musiq(data_input).mean().item())

        uiqm_avg_out.update(uiqm_out, 1)
        uciqe_avg_out.update(uciqe_out, 1)
        musiq_avg_out.update(musiq_out, 1)

        uiqm_avg_in.update(uiqm_in, 1)
        uciqe_avg_in.update(uciqe_in, 1)
        musiq_avg_in.update(musiq_in, 1)

        # Save result
        temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
        temp_res = np.clip(temp_res, 0, 1)
        temp_res = (temp_res * 255).astype(np.uint8)
        Image.fromarray(temp_res).save(os.path.join(save_path, name))

        tbar.set_description(
            f"Test: {name} | "
            f"PSNR(avg): {test_psnr.avg:.4f}, SSIM(avg): {test_ssim.avg:.4f} | "
            f"OUT UIQM(avg): {uiqm_avg_out.avg:.3f}, UCIQE(avg): {uciqe_avg_out.avg:.3f}, MUSIQ(avg): {musiq_avg_out.avg:.3f}"
        )
        tbar.set_postfix({
            "IN_UIQM(avg)": f"{uiqm_avg_in.avg:.3f}",
            "IN_UCIQE(avg)": f"{uciqe_avg_in.avg:.3f}",
            "IN_MUSIQ(avg)": f"{musiq_avg_in.avg:.3f}",
        })

print("\nfinished!")
print(f"Saved enhanced images to: {save_path}")
print(f"Final PSNR : {test_psnr.avg:.6f}")
print(f"Final SSIM : {test_ssim.avg:.6f}")

print("\nNo-reference metrics (OUTPUT):")
print(f"UIQM : {uiqm_avg_out.avg:.6f}")
print(f"UCIQE: {uciqe_avg_out.avg:.6f}")
print(f"MUSIQ: {musiq_avg_out.avg:.6f}")

print("\nNo-reference metrics (INPUT) [for comparison]:")
print(f"UIQM : {uiqm_avg_in.avg:.6f}")
print(f"UCIQE: {uciqe_avg_in.avg:.6f}")
print(f"MUSIQ: {musiq_avg_in.avg:.6f}")
