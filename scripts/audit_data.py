"""
CataSaarthi - Milestone 2 / Task 8: raw data audit for ODIR-5K.

READ-ONLY. Measures the REAL inventory and label distribution of the
downloaded dataset straight from disk. Nothing is assumed or hard-coded -
every number printed is counted from the files. This is the honest basis
for deriving our binary Normal vs Suspected-Cataract labels next.

Run from the project root:  python scripts/audit_data.py
"""
from pathlib import Path
import sys
import yaml
import pandas as pd

# --- locate project + config (config.yaml is the single source of truth for paths) ---
ROOT = Path(__file__).resolve().parents[1]          # D:\CataSaarthi
cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
data_root = Path(cfg["data"]["root"])
raw = data_root / cfg["data"]["raw_subdir"]

print("PROJECT ROOT :", ROOT)
print("DATA ROOT    :", data_root)
print("RAW DIR      :", raw, "| exists:", raw.exists())
print("=" * 72)

# --- 1. what is physically in raw/ ---
print("TOP-LEVEL CONTENTS OF raw/:")
for p in sorted(raw.iterdir()):
    tag = "DIR " if p.is_dir() else "FILE"
    size = "" if p.is_dir() else f"{p.stat().st_size/1e6:.2f} MB"
    print(f"   [{tag}] {p.name} {size}")
print("=" * 72)

# --- 2. count images in each image folder we find ---
IMG_EXT = {".jpg", ".jpeg", ".png"}
def count_images(folder: Path) -> int:
    return sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMG_EXT)

for sub in ["preprocessed_images", "ODIR-5K"]:
    folder = raw / sub
    if not folder.exists():
        print(f"[{sub}] FOLDER NOT FOUND")
        continue
    print(f"[{sub}] total images: {count_images(folder)}")
    for child in sorted(folder.iterdir()):
        if child.is_dir():
            print(f"     - {child.name}/  ({count_images(child)} images)")
print("=" * 72)

# --- 3. load full_df.csv and describe it ---
csv_path = raw / "full_df.csv"
if not csv_path.exists():
    print("full_df.csv NOT found. Other data files present:")
    for f in list(raw.rglob("*.csv")) + list(raw.rglob("*.xlsx")):
        print("   ", f)
    sys.exit(0)

df = pd.read_csv(csv_path)
print(f"full_df.csv : {df.shape[0]} rows x {df.shape[1]} columns")
print("COLUMNS     :", list(df.columns))
print("-" * 72)
print("FIRST 3 ROWS (transposed):")
with pd.option_context("display.max_columns", None, "display.width", 220):
    print(df.head(3).T)
print("=" * 72)

# --- 4. REAL label distribution for the 8 ODIR targets, if present ---
TARGETS = ["N", "D", "G", "C", "A", "H", "M", "O"]
present = [c for c in TARGETS if c in df.columns]
if present:
    print("PER-ROW LABEL DISTRIBUTION (real counts):")
    for c in present:
        pos = int(df[c].sum())
        print(f"   {c}: {pos:>5}  ({100*pos/len(df):5.1f}%)")
    if "C" in present:
        print(f"\n   >>> CATARACT (C) positives: {int(df['C'].sum())} of {len(df)} rows")

for col in ["labels", "target"]:
    if col in df.columns:
        print("-" * 72)
        print(f"VALUE COUNTS of {col!r} (top 20):")
        print(df[col].value_counts().head(20).to_string())
print("=" * 72)

# --- 5. patient vs eye/image counts (data-leakage relevant) ---
for c in df.columns:
    if c.lower() == "id":
        print(f"Unique {c!r} (patients?): {df[c].nunique()}")
if "filename" in df.columns:
    print(f"Unique 'filename': {df['filename'].nunique()}  (of {len(df)} rows)")
print("=" * 72)
print("AUDIT COMPLETE - copy this whole output back to Claude.")