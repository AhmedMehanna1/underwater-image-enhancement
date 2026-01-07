import os
from pathlib import Path

import torch
import torch.fx
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2
import pyiqa

# my imports
from model import AIMnet
from dataset_all import TestData


# -----------------------------
# Utils: save tensor image
# -----------------------------
def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    """
    x: (1,3,H,W) or (3,H,W), assumed in [0,1]
    returns uint8 RGB (H,W,3)
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return x


# -----------------------------
# UIQM implementation (UICM + UISM + UIConM)
# -----------------------------
def _alpha_trim(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    x = np.sort(x.flatten())
    n = x.size
    k = int(alpha * n)
    if n - 2 * k <= 0:
        return x
    return x[k: n - k]


def _uicm(rgb: np.ndarray) -> float:
    # rgb uint8 [0..255]
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

    # Standard UICM form used in common UIQM implementations
    return np.sqrt(mu_rg * mu_rg + mu_yb * mu_yb) + 0.3 * np.sqrt(sigma_rg * sigma_rg + sigma_yb * sigma_yb)


def _eme(gray: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    """
    Enhancement Measure Estimation (EME) computed on gray float32 image [0..1]
    """
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
    if count == 0:
        return 0.0
    return eme_sum / count


def _uism(rgb: np.ndarray) -> float:
    """
    Underwater Image Sharpness Measure (UISM) - common implementation:
    compute Sobel gradient magnitude per channel then EME over each, weighted.
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    weights = [0.299, 0.587, 0.114]  # approx luminance weights
    uism = 0.0
    for c, w in enumerate(weights):
        ch = rgb_f[:, :, c]
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        uism += w * _eme(gmag, block_size=8)
    return float(uism)


def _uiconm(rgb: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    """
    Underwater Image Contrast Measure (UIConM) - common block-based contrast measure.
    """
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
            # log contrast in block
            s += np.log((bmax - bmin + eps) / (bmax + bmin + eps) + 1.0)
            count += 1
    if count == 0:
        return 0.0
    return float(s / count)


def uiqm(rgb_uint8: np.ndarray) -> float:
    """
    UIQM = 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM
    """
    uicm = _uicm(rgb_uint8)
    uism = _uism(rgb_uint8)
    uiconm = _uiconm(rgb_uint8)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


# -----------------------------
# UCIQE implementation
# -----------------------------
def uciqe(rgb_uint8: np.ndarray) -> float:
    """
    Common UCIQE implementation in Lab space:
    UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s

    - sigma_c: std of chroma in Lab
    - con_l: contrast of L (max-min)
    - mu_s: mean saturation (C / sqrt(C^2 + L^2))
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    # convert RGB->Lab (OpenCV expects BGR)
    bgr = rgb[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0] / 255.0  # normalize [0..1] (approx)
    a = (lab[:, :, 1] - 128.0) / 255.0
    b = (lab[:, :, 2] - 128.0) / 255.0

    C = np.sqrt(a * a + b * b)
    sigma_c = float(np.std(C))
    con_l = float(np.max(L) - np.min(L))

    # saturation proxy used in many UCIQE implementations
    mu_s = float(np.mean(C / (np.sqrt(C * C + L * L) + 1e-6)))

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


# -----------------------------
# Main test
# -----------------------------
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    bz = 1
    model_root = "model/ckpt/model_best_student.pth"
    input_root = "test/uieb-unpaired"
    save_path = "test/uieb-unpaired/enhanced"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(model_root, map_location="cpu")

    # dataset
    Mydata_ = TestData(input_root, 256)
    data_load = data.DataLoader(Mydata_, batch_size=bz, shuffle=False, num_workers=0, pin_memory=True)

    # model
    model = AIMnet().cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.load_state_dict(checkpoint)
    model.eval()

    # MUSIQ metric
    # For evaluation scores (not training), use as_loss=False.
    musiq = pyiqa.create_metric("musiq", as_loss=False).cuda()
    musiq.eval()
    for p in musiq.parameters():
        p.requires_grad_(False)

    # running stats
    n = 0
    uiqm_sum = 0.0
    uciqe_sum = 0.0
    musiq_sum = 0.0

    print("START!")
    pbar = tqdm(data_load, ncols=130, desc="Testing")

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            # Your TestData returns: data_input, data_la
            data_input, data_la = batch

            data_input = Variable(data_input).cuda(non_blocking=True)
            data_la = Variable(data_la).cuda(non_blocking=True)

            result, _ = model(data_input, data_la)

            # save output
            out_rgb = tensor_to_uint8_rgb(result)  # uint8 RGB
            out_pil = Image.fromarray(out_rgb)

            # try to get original filename from dataset
            # common patterns: dataset.A_paths[idx] or dataset.A_path / dataset.A
            name = None
            if hasattr(Mydata_, "A_paths"):
                name = Path(Mydata_.A_paths[idx]).name
            elif hasattr(Mydata_, "A"):
                # some repos store file list in Mydata_.A
                try:
                    name = Path(Mydata_.A[idx]).name
                except Exception:
                    name = f"{idx:05d}.png"
            else:
                name = f"{idx:05d}.png"

            out_file = Path(save_path) / name
            out_pil.save(str(out_file))

            # compute UIQM & UCIQE (CPU)
            uiqm_v = uiqm(out_rgb)
            uciqe_v = uciqe(out_rgb)

            # compute MUSIQ (GPU, expects torch tensor in [0,1], shape (N,3,H,W))
            musiq_v = float(musiq(result).mean().item())

            # update running averages
            n += 1
            uiqm_sum += uiqm_v
            uciqe_sum += uciqe_v
            musiq_sum += musiq_v

            pbar.set_postfix({
                "UIQM": f"{uiqm_v:.3f}",
                "UCIQE": f"{uciqe_v:.3f}",
                "MUSIQ": f"{musiq_v:.3f}",
                "UIQM_avg": f"{uiqm_sum/n:.3f}",
                "UCIQE_avg": f"{uciqe_sum/n:.3f}",
                "MUSIQ_avg": f"{musiq_sum/n:.3f}",
            })

    print("\nDONE.")
    print(f"Saved enhanced images to: {save_path}")
    print(f"Avg UIQM : {uiqm_sum/n:.6f}")
    print(f"Avg UCIQE: {uciqe_sum/n:.6f}")
    print(f"Avg MUSIQ: {musiq_sum/n:.6f}")


if __name__ == "__main__":
    main()

