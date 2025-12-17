import os
from glob import glob
from os.path import join, basename
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def luminance_estimation(img: Image.Image) -> np.ndarray:
    sigma_list = [15, 60, 90]
    img = np.uint8(np.array(img))

    illuminance = np.ones_like(img).astype(np.float32)
    for sigma in sigma_list:
        illuminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8)
        illuminance1 = np.clip(illuminance1, 0, 255)
        illuminance = illuminance + illuminance1

    illuminance = illuminance / 3.0
    L = (illuminance - np.min(illuminance)) / (np.max(illuminance) - np.min(illuminance) + 1e-6)
    return np.uint8(L * 255)

def process_one(path: str, out_dir: str):
    img = Image.open(path).convert("RGB")
    L = luminance_estimation(img)
    out_path = join(out_dir, basename(path))
    Image.fromarray(L).save(out_path)
    return path

def main():
    input_dir = "data/val/input"
    result_dir = "data/val/LA"
    os.makedirs(result_dir, exist_ok=True)

    paths = glob(join(input_dir, "*.*"))
    if not paths:
        print(f"No images found in: {input_dir}")
        return

    cpu = os.cpu_count() or 4
    max_workers = max(1, cpu - 1) 

    errors = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, p, result_dir) for p in paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating LA"):
            try:
                fut.result()
            except Exception as e:
                errors.append(repr(e))

    print("finished!")
    if errors:
        print(f"Errors: {len(errors)} (showing 5)")
        for e in errors[:5]:
            print(e)

if __name__ == "__main__":
    main()
