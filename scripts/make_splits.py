"""
CataSaarthi - Milestone 2 / Task 10: patient-level train/val/test split.

Leakage rule: BOTH eyes of a patient must stay in the SAME split. We split at
the PATIENT level (grouping guaranteed), stratified by whether the patient has
any cataract eye, so each split keeps a representative share of the rare
positive class. Seed and split sizes come from configs/config.yaml.

Output: <data_root>/processed/splits.csv   (per-eye rows + a 'split' column)
Run:    python scripts/make_splits.py
"""
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
processed = Path(cfg["data"]["root"]) / cfg["data"]["processed_subdir"]
seed      = int(cfg["project"]["seed"])
val_size  = float(cfg["split"]["val_size"])
test_size = float(cfg["split"]["test_size"])

df = pd.read_csv(processed / "labels_binary.csv")

# --- 1. patient-level table + stratification label (has any cataract eye) ---
pat = (df.groupby("patient_id")["label"].max()
         .rename("has_cataract").reset_index())
print(f"patients: {len(pat)}   cataract-patients: {int(pat.has_cataract.sum())}")

# --- 2. split PATIENTS: hold out test first, then carve val from the rest ---
trainval, test = train_test_split(
    pat, test_size=test_size, stratify=pat["has_cataract"], random_state=seed)
val_rel = val_size / (1.0 - test_size)          # val as a fraction of trainval
train, val = train_test_split(
    trainval, test_size=val_rel, stratify=trainval["has_cataract"], random_state=seed)

split_of = {}
for pid in train["patient_id"]: split_of[pid] = "train"
for pid in val["patient_id"]:   split_of[pid] = "val"
for pid in test["patient_id"]:  split_of[pid] = "test"
df["split"] = df["patient_id"].map(split_of)

# --- 3. report per-split eye counts + cataract share ---
print("=" * 62)
print(f"{'split':6}{'eyes':>7}{'normal':>8}{'cataract':>10}{'cat%':>7}{'patients':>10}")
for s in ["train", "val", "test"]:
    d = df[df.split == s]
    n_cat, n_norm = int((d.label == 1).sum()), int((d.label == 0).sum())
    print(f"{s:6}{len(d):>7}{n_norm:>8}{n_cat:>10}{100*n_cat/max(len(d),1):>6.1f}%{d.patient_id.nunique():>10}")
print("=" * 62)

# --- 4. save ---
dest = processed / "splits.csv"
df.to_csv(dest, index=False)
print(f"saved -> {dest}")
print("Test set is LOCKED: use ONLY for the final report, never for tuning.")