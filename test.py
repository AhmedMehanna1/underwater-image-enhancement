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

from utils import tensor_to_uint8_rgb, uiqm, uciqe


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

