from pathlib import Path
import random
import shutil
import os

def make_subset(
    input_dir: str,
    gt_dir: str,
    out_dir: str,
    k: int = 800,
    seed: int = 2022,
    exts=(".png", ".jpg", ".jpeg", ".bmp")
):
    input_dir = Path(input_dir)
    gt_dir = Path(gt_dir)
    out_dir = Path(out_dir)

    out_in = out_dir / "input"
    out_gt = out_dir / "GT"
    out_in.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    # collect filenames that exist in both (paired by name)
    in_names = {p.name for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    gt_names = {p.name for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    common = sorted(in_names & gt_names)

    if len(common) < k:
        raise RuntimeError(f"Only {len(common)} paired files found, cannot sample {k}.")

    rng = random.Random(seed)
    rng.shuffle(common)
    chosen = common[:k]

    # Choose one: copy or symlink
    USE_SYMLINK = False  # set True if you want symlinks instead of copying

    for name in chosen:
        src_in = input_dir / name
        src_gt = gt_dir / name
        dst_in = out_in / name
        dst_gt = out_gt / name

        if USE_SYMLINK:
            # remove if exists
            if dst_in.exists(): dst_in.unlink()
            if dst_gt.exists(): dst_gt.unlink()
            os.symlink(src_in, dst_in)
            os.symlink(src_gt, dst_gt)
        else:
            shutil.copy2(src_in, dst_in)
            shutil.copy2(src_gt, dst_gt)

    print(f"Done. Created {k} pairs at: {out_dir}")
    print(f"Example pair: {out_in / chosen[0]}  <->  {out_gt / chosen[0]}")

if __name__ == "__main__":
    make_subset(
        input_dir="./data2/uwcnn_merged/input",
        gt_dir="./data2/uwcnn_merged/GT",
        out_dir="./data2/uwcnn_800",
        k=800,
        seed=2022
    )
