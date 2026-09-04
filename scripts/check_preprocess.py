"""
CataSaarthi - Milestone 2 / Task 11 verification.

Exercises the REAL preprocessing pipeline end-to-end:
  * indexes the original images,
  * builds the train/val/test datasets,
  * pushes a few eyes through the transforms and prints tensor stats,
  * saves a de-normalized montage (2 normal + 2 cataract) to outputs/ so the
    border-crop + resize can be eyeballed.

Run from the project root:  python scripts/check_preprocess.py
"""
from pathlib import Path
import numpy as np
import torch
import cv2

from catasaarthi.config import load_config
from catasaarthi.dataset import FundusDataset, build_image_index
from catasaarthi.preprocess import IMAGENET_MEAN, IMAGENET_STD

cfg = load_config()
size = int(cfg["image"]["size"])

index = build_image_index(cfg)
print(f"originals indexed: {len(index)} images")

datasets = {}
for split in ["train", "val", "test"]:
    datasets[split] = FundusDataset(split, cfg=cfg, image_index=index)
    print(f"  {split:5}: {len(datasets[split]):5d} eyes")
total = sum(len(d) for d in datasets.values())
print(f"  total: {total} eyes  (expected 3166)")
print("=" * 60)

# --- tensor sanity on the deterministic val transform ---
val = datasets["val"]
labels = val.df["label"].values
pick = list(np.where(labels == 0)[0][:2]) + list(np.where(labels == 1)[0][:2])

mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
tiles = []
for i in pick:
    x, y = val[int(i)]
    print(f"val[{int(i):4d}] shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.min():.2f} max={x.max():.2f} mean={x.mean():.2f} label={y}")
    img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.putText(img, f"label={y}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    tiles.append(img)

# --- confirm the train (augmented) transform yields the same shape ---
xt, yt = datasets["train"][0]
print(f"train[0] shape={tuple(xt.shape)} label={yt}  (augmented)")
print("=" * 60)

out_dir = Path(cfg["data"]["root"]) / cfg["data"]["outputs_subdir"]
out_dir.mkdir(parents=True, exist_ok=True)
dest = out_dir / "preprocess_samples.png"
cv2.imwrite(str(dest), np.hstack(tiles))
print(f"montage (2 normal + 2 cataract) saved -> {dest}")
print("DONE - copy this output back, and open the PNG to eyeball the crop.")