import cv2
import torch
import torch.fx
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# recommend
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)  
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

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
    eme_sum = 0.0
    count = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            eme_sum += np.log((bmax + eps) / (bmin + eps))
            count += 1
    return 0.0 if count == 0 else eme_sum / count

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
    s = 0.0
    count = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            if block.size == 0:
                continue
            bmax = float(np.max(block))
            bmin = float(np.min(block))
            s += np.log((bmax - bmin + eps) / (bmax + bmin + eps) + 1.0)
            count += 1
    return 0.0 if count == 0 else float(s / count)

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
    UCIQE in the commonly reported (<1) range.
    - Uses OpenCV Lab, then normalizes:
      L in [0,1], a/b in roughly [-0.5,0.5]
    """
    rgb = rgb_uint8.astype(np.float32) / 255.0
    bgr = rgb[:, :, ::-1]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # OpenCV: L,a,b in [0..255]
    L = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 255.0
    b = (lab[:, :, 2] - 128.0) / 255.0

    C = np.sqrt(a * a + b * b)
    sigma_c = float(np.std(C))
    con_l = float(np.max(L) - np.min(L))
    mu_s = float(np.mean(C / (np.sqrt(C * C + L * L) + 1e-6)))

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)
