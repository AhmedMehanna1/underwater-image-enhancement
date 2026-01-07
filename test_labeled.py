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
from utils import AverageMeter, compute_psnr_ssim, to_psnr


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
# UIQM + UCIQE helpers
# -----------------------------
def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    """
    x: (1,3,H,W) or (3,H,W) in [0,1]
    -> uint8 RGB (H,W,3)
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return x


def _alpha_trim(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    x = np.sort(x.flatten())
    n = x.size
    k = int(alpha * n)
    if n - 2 * k <= 0:
        return x
    return x[k: n - k]


def _uicm(rgb: np.ndarray) -> float:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    rg = r - g
    yb = 0.5 * (r + g) - b

    rg_t = _alpha_trim(rg, 0.1)
    yb_t = _alpha_trim(yb, 0.1)

    mu_rg = float(np.mean(rg_t))
    mu_yb = float(np.mean(yb_t))
    sigma_rg = float(np.std(rg_t))
    sigma_yb = float(np.std(yb_t))

    return np.sqrt(mu_rg * mu_rg + mu_yb * mu_yb) + 0.3 * np.sqrt(sigma_rg * sigma_rg + sigma_yb * sigma_yb)


def _eme(gray: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    h, w = gray.shape
    bh = max(1, h // block_size)
    bw = max(1, w // block_size)

    eme_sum = 0.0
    count = 0
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = gray[i:i + bh, j:j + bw]
            if block.size == 0:
                continue
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            eme_sum += np.log((bmax + eps) / (bmin + eps))
            count += 1
    return eme_sum / count if count else 0.0


def _uism(rgb: np.ndarray) -> float:
    rgb_f = rgb.astype(np.float32) / 255.0
    weights = [0.299, 0.587, 0.114]
    uism = 0.0
    for c, w in enumerate(weights):
        ch = rgb_f[:, :, c]
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        uism += w * _eme(gmag, block_size=8)
    return float(uism)


def _uiconm(rgb: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    bh = max(1, h // block_size)
    bw = max(1, w // block_size)

    s = 0.0
    count = 0
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = gray[i:i + bh, j:j + bw]
            if block.size == 0:
                continue
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            s += np.log((bmax - bmin + eps) / (bmax + bmin + eps) + 1.0)
            count += 1
    return float(s / count) if count else 0.0


def uiqm(rgb_uint8: np.ndarray) -> float:
    return float(0.0282 * _uicm(rgb_uint8) + 0.2953 * _uism(rgb_uint8) + 3.5753 * _uiconm(rgb_uint8))


def uciqe(rgb_uint8: np.ndarray) -> float:
    rgb = rgb_uint8.astype(np.float32) / 255.0
    bgr = rgb[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 255.0
    b = (lab[:, :, 2] - 128.0) / 255.0

    C = np.sqrt(a * a + b * b)
    sigma_c = float(np.std(C))
    con_l = float(np.max(L) - np.min(L))
    mu_s = float(np.mean(C / (np.sqrt(C * C + L * L) + 1e-6)))

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


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
