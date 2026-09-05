"""
CataSaarthi - decision-threshold tuning for SCREENING sensitivity.

The baseline saves best.pt by val AUPRC, but the default 0.5 cutoff can be
too conservative for screening (misses true cataracts). Here we load that
same checkpoint, score the VALIDATION set once, and sweep the decision
threshold to find the operating point that meets a target sensitivity
(recall on cataract) while keeping specificity as high as possible.

Integrity: tuning uses VALIDATION only. The test set is never loaded here.
Outputs: <work>/outputs/threshold_sweep.csv  and  <work>/outputs/threshold.json
Run: python scripts/tune_threshold.py --target-recall 0.90
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from catasaarthi.config import load_config
from catasaarthi.dataset import FundusDataset, build_image_index


def build_model(backbone: str, num_classes: int = 2):
    name = backbone.lower()
    if name == "resnet18":
        net = models.resnet18(weights=None)
    elif name == "resnet50":
        net = models.resnet50(weights=None)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


@torch.no_grad()
def get_val_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def metrics_at(y_true, y_prob, t):
    y_pred = (y_prob >= t).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"threshold": round(float(t), 3), "recall": round(recall, 3),
            "specificity": round(specificity, 3), "precision": round(precision, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.90,
                    help="minimum cataract sensitivity the chosen threshold must meet")
    ap.add_argument("--min-threshold", type=float, default=0.0,
                    help="don't choose a threshold below this (avoids over-tuning to a "
                         "single noisy validation eye at very low cutoffs)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models_dir = Path(cfg["data"]["root"]) / cfg["data"]["models_subdir"]
    out_dir = Path(cfg["data"]["root"]) / cfg["data"]["outputs_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # weights_only=False: this is OUR checkpoint saved by train.py in the same
    # pipeline (trusted). It stores a metrics dict with numpy scalars, which the
    # torch>=2.6 strict loader (weights_only=True) refuses to unpickle.
    ckpt = torch.load(models_dir / "best.pt", map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", cfg["model"]["backbone"])
    model = build_model(backbone).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded best.pt  backbone={backbone}  epoch={ckpt.get('epoch')}  device={device}")

    index = build_image_index(cfg)
    val_ds = FundusDataset("val", cfg=cfg, image_index=index)
    val_ld = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False, num_workers=int(cfg["train"]["num_workers"]))
    y_true, y_prob = get_val_probs(model, val_ld, device)
    print(f"val eyes={len(y_true)}  cataract={int(y_true.sum())}")

    rows = [metrics_at(y_true, y_prob, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)]

    print("\n thr  recall  spec  prec   TP FP FN TN")
    for r in rows:
        print(f"{r['threshold']:.2f}  {r['recall']:.3f}  {r['specificity']:.3f}  "
              f"{r['precision']:.3f}  {r['tp']:3d}{r['fp']:3d}{r['fn']:3d}{r['tn']:4d}")

    # meet the sensitivity floor AND respect the minimum-threshold guard, then take
    # the HIGHEST such threshold (best specificity for the required sensitivity).
    ok = [r for r in rows
          if r["recall"] >= args.target_recall and r["threshold"] >= args.min_threshold]
    chosen = max(ok, key=lambda r: r["threshold"]) if ok else max(rows, key=lambda r: r["recall"])
    
    import csv
    with open(out_dir / "threshold_sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    (out_dir / "threshold.json").write_text(json.dumps(
        {"target_recall": args.target_recall, "chosen": chosen, "backbone": backbone,
         "epoch": ckpt.get("epoch")}, indent=2))

    met = "met" if ok else "NOT met - using most sensitive"
    print(f"\ntarget recall {args.target_recall} {met}")
    print(f"CHOSEN threshold = {chosen['threshold']}  "
          f"(recall={chosen['recall']}, specificity={chosen['specificity']}, "
          f"precision={chosen['precision']})")
    print(f"saved -> {out_dir/'threshold_sweep.csv'} and threshold.json")


if __name__ == "__main__":
    main()