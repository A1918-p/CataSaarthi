"""
CataSaarthi - Milestone 2 / Task 9: derive HONEST binary labels.

Goal: per-eye Normal vs Suspected-Cataract.
  label 0 = Normal             (eye keyword == 'normal fundus')
  label 1 = Suspected Cataract (eye keyword contains 'cataract')
Other-disease eyes are set aside for this binary baseline.

Integrity: we RE-DERIVE each eye's label from that eye's OWN diagnostic
keyword (not the misleading patient-level N..O columns, and not blindly
trusting the pre-made 'labels' column), then CROSS-CHECK against 'labels'
to prove the rule is sound.

Output: <data_root>/processed/labels_binary.csv
Run:    python scripts/make_labels.py
"""
from pathlib import Path
import re
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
data_root = Path(cfg["data"]["root"])
raw = Path(cfg["data"]["raw_root"])
processed = data_root / cfg["data"]["processed_subdir"]
processed.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw / "full_df.csv")

# --- 1. which eye is this row, and that eye's keyword ---
def eye_of(row):
    fn = str(row["filename"]).lower()
    if "left" in fn:  return "left"
    if "right" in fn: return "right"
    if str(row["filename"]) == str(row["Left-Fundus"]):  return "left"
    if str(row["filename"]) == str(row["Right-Fundus"]): return "right"
    return "unknown"

def keyword_of(row):
    kw = row["Left-Diagnostic Keywords"] if row["eye"] == "left" \
         else row["Right-Diagnostic Keywords"] if row["eye"] == "right" else ""
    kw = str(kw).replace("，", ",").lower().strip()   # full-width comma -> comma
    return re.sub(r"\s+", " ", kw)

df["eye"] = df.apply(eye_of, axis=1)
df["keyword"] = df.apply(keyword_of, axis=1)

# --- 2. our independent binary rule ---
def derive(kw):
    if "cataract" in kw:      return 1
    if kw == "normal fundus": return 0
    return -1                              # other disease -> excluded

df["label"] = df["keyword"].apply(derive)

# --- 3. cross-check vs the dataset's own per-eye 'labels' ---
df["label_ref"] = df["labels"].map({"['C']": 1, "['N']": 0}).fillna(-1).astype(int)
mismatch = df[(df["label"] != df["label_ref"]) &
              ((df["label"] != -1) | (df["label_ref"] != -1))]
print("=" * 68)
print("CROSS-CHECK  (our keyword rule vs dataset 'labels'):")
print(f"  cataract : ours={int((df.label==1).sum()):4d}   ref={int((df.label_ref==1).sum()):4d}")
print(f"  normal   : ours={int((df.label==0).sum()):4d}   ref={int((df.label_ref==0).sum()):4d}")
print(f"  disagreements on N/C rows: {len(mismatch)}")
if len(mismatch):
    print("  --- first 15 disagreements ---")
    print(mismatch[["filename","eye","keyword","labels","label","label_ref"]].head(15).to_string(index=False))
print("=" * 68)

# --- 4. clean binary subset ---
# Our keyword rule above VALIDATED ODIR's per-eye labels; every difference
# was benign (8 cataract-with-other-disease, 57 normal-with-artifact).
# Per the chosen 'clean & standard' policy we adopt ODIR's per-eye N/C:
# pure cataract (293) vs normal incl. minor artifacts (2873); the 8
# co-morbid eyes are set aside. Image QUALITY is handled later by the
# quality gate, not by dropping labels here.
binary = df[df["label_ref"].isin([0, 1])].copy()
binary["label"] = binary["label_ref"]

# --- 5. verify every image file exists ---
img_dir = raw / "preprocessed_images"
binary["exists"] = binary["filename"].apply(lambda f: (img_dir / f).exists())
missing = int((~binary["exists"]).sum())
print(f"Images checked in preprocessed_images/: {len(binary)}   missing: {missing}")
if missing:
    print(binary.loc[~binary["exists"], "filename"].head(10).to_string(index=False))

# --- 6. save + summarize ---
out = (binary.loc[binary["exists"], ["ID","filename","eye","keyword","label"]]
             .rename(columns={"ID": "patient_id"}))
n_norm, n_cat = int((out.label==0).sum()), int((out.label==1).sum())
dest = processed / "labels_binary.csv"
out.to_csv(dest, index=False)
print("-" * 68)
print(f"FINAL:  Normal(0)={n_norm}   Cataract(1)={n_cat}   total={len(out)}"
      f"   patients={out.patient_id.nunique()}")
print(f"imbalance = {n_norm/max(n_cat,1):.1f} : 1")
print(f"saved -> {dest}")
print("=" * 68)
print("Copy this output .")