import torch
import torch.nn as nn
import torch.nn.functional as F


class ChromaLoss(nn.Module):
    """
    L1 on chroma channels (Cb, Cr). Helps fix color casts while preserving texture PSNR.
    """
    def __init__(self, weight_cb=1.0, weight_cr=1.0):
        super().__init__()
        self.wcb = weight_cb
        self.wcr = weight_cr

    def __rgb_to_ycbcr(self, x: torch.Tensor):
        """
        x: (B,3,H,W) in [0,1]
        returns y, cb, cr in roughly [0,1]
        """
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return y, cb, cr

    def forward(self, pred, gt):
        _, cb_p, cr_p = self.__rgb_to_ycbcr(pred)
        _, cb_g, cr_g = self.__rgb_to_ycbcr(gt)
        return self.wcb * F.l1_loss(cb_p, cb_g) + self.wcr * F.l1_loss(cr_p, cr_g)

class SaturationLoss(nn.Module):
    """
    Match saturation maps between pred and gt:
    sat = max(rgb) - min(rgb)
    """
    def forward(self, pred, gt):
        mx_p, _ = pred.max(dim=1, keepdim=True)
        mn_p, _ = pred.min(dim=1, keepdim=True)
        sat_p = mx_p - mn_p

        mx_g, _ = gt.max(dim=1, keepdim=True)
        mn_g, _ = gt.min(dim=1, keepdim=True)
        sat_g = mx_g - mn_g

        return F.l1_loss(sat_p, sat_g)

class GrayWorldLoss(nn.Module):
    """
    Encourage channel means to be closer (reduces strong casts).
    """
    def forward(self, pred):
        m = pred.mean(dim=[2, 3])  # (B,3)
        r, g, b = m[:, 0], m[:, 1], m[:, 2]
        return ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        # self.L1 = nn.L1Loss()
        self.Charbonnier = CharbonnierLoss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):
        L2_temp = 0.2 * self.L2(xs, ys)
        # L1_temp = 0.8 * self.L1(xs, ys)
        # L_total = L1_temp + L2_temp
        charbonnier_temp = 0.8 * self.Charbonnier(xs, ys)
        L_total = charbonnier_temp + L2_temp
        return L_total


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, res, gt):
        res = (res + 1.0) * 127.5
        gt = (gt + 1.0) * 127.5
        r_mean = (res[:, 0, :, :] + gt[:, 0, :, :]) / 2.0
        r = res[:, 0, :, :] - gt[:, 0, :, :]
        g = res[:, 1, :, :] - gt[:, 1, :, :]
        b = res[:, 2, :, :] - gt[:, 2, :, :]
        p_loss_temp = (((512 + r_mean) * r * r) / 256) + 4 * g * g + (((767 - r_mean) * b * b) / 256)
        p_loss = torch.mean(torch.sqrt(p_loss_temp + 1e-8)) / 255.0
        return p_loss


class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        # ImageNet normalization constants (register as buffers so they move with .cuda()/.to())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Your pipeline is [0,1]. Clamp for safety then normalize for VGG.
        x = x.clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        # Normalize both inputs as VGG expects ImageNet-normalized tensors
        dehaze_n = self._imagenet_norm(dehaze)
        gt_n = self._imagenet_norm(gt)

        dehaze_features = self.output_features(dehaze_n)
        gt_features = self.output_features(gt_n)

        loss = 0.0
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss = loss + F.mse_loss(dehaze_feature, gt_feature)

        return loss / len(dehaze_features)


# Charbonnier loss
class CharLoss(nn.Module):
    def __init__(self):
        super(CharLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(pred, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

