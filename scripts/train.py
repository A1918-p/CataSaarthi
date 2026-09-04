"""
CataSaarthi - Milestone 3: baseline training (device-agnostic, config-driven).

Screening baseline: a pretrained CNN with a 2-class head (Normal vs Suspected
Cataract), trained with a CLASS-WEIGHTED loss for the ~9.8:1 imbalance. After
each epoch we report val AUROC / AUPRC / cataract-recall + confusion matrix and
save the checkpoint with the best val AUPRC. The LOCKED test set is never used.

Defaults come from configs/config.yaml; CLI flags override them.

Fast smoke test (local CPU):
    python scripts/train.py --backbone resnet18 --epochs 1 --max-batches 15
Real run (cloud GPU):
    python scripts/train.py --backbone resnet50 --epochs 15
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from catasaarthi.config import load_config
from catasaarthi.dataset import FundusDataset, build_image_index
from catasaarthi.metrics import compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(backbone: str, pretrained: bool, num_classes: int = 2):
    name = backbone.lower()
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"unsupported backbone: {backbone}")
    net.fc = nn.Linear(net.fc.in_features, num_classes)   # new head for 2 classes
    return net


def class_weights(labels, device):
    """Inverse-frequency weights so the rare cataract class isn't ignored."""
    labels = np.asarray(labels)
    total = len(labels)
    w = [total / (2.0 * max(int((labels == c).sum()), 1)) for c in (0, 1)]
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]   # P(cataract)
        probs.append(p.cpu())
        ys.append(y)
    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()
    return compute_metrics(ys, probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="cap train batches per epoch (0 = no cap; use for a fast smoke test)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    backbone = args.backbone or cfg["model"]["backbone"]
    pretrained = bool(cfg["model"]["pretrained"])
    epochs = args.epochs or int(cfg["train"]["epochs"])
    batch_size = args.batch_size or int(cfg["train"]["batch_size"])
    lr = args.lr or float(cfg["train"]["lr"])
    num_workers = int(cfg["train"]["num_workers"])
    monitor = cfg["train"].get("monitor", "auprc")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  backbone={backbone}  epochs={epochs}  "
          f"batch_size={batch_size}  lr={lr}")

    index = build_image_index(cfg)
    train_ds = FundusDataset("train", cfg=cfg, image_index=index)
    val_ds = FundusDataset("val", cfg=cfg, image_index=index)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(backbone, pretrained).to(device)
    weights = class_weights(train_ds.df["label"].values, device)
    print(f"class weights (normal, cataract) = ({weights[0]:.3f}, {weights[1]:.3f})")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    models_dir = Path(cfg["data"]["root"]) / cfg["data"]["models_subdir"]
    out_dir = Path(cfg["data"]["root"]) / cfg["data"]["outputs_subdir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    best, log_rows = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for bi, (x, y) in enumerate(train_ld):
            if args.max_batches and bi >= args.max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
        train_loss = running / max(n, 1)

        m = evaluate(model, val_ld, device)
        print(f"epoch {epoch:02d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_AUROC={m['auroc']:.3f}  val_AUPRC={m['auprc']:.3f}  "
              f"cataract_recall={m['recall_cataract']:.3f}  "
              f"[TP{m['tp']} FP{m['fp']} FN{m['fn']} TN{m['tn']}]")
        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 4), **m})

        score = m.get(monitor, float("nan"))
        if score == score and score > best:          # score==score skips NaN
            best = score
            torch.save({"model_state": model.state_dict(), "backbone": backbone,
                        "epoch": epoch, "metrics": m, "seed": seed},
                       models_dir / "best.pt")
            print(f"    saved best.pt (val {monitor}={best:.3f})")

    with open(out_dir / "train_log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        wr.writeheader()
        wr.writerows(log_rows)
    print(f"log -> {out_dir / 'train_log.csv'}")
    print("DONE - copy this output back to Claude.")


if __name__ == "__main__":
    main()