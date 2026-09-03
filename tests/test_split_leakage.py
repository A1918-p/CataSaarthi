"""
Leakage + integrity tests for the patient-level split (Task 10).
Run:  pytest -q
"""
from pathlib import Path
import yaml
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
processed = Path(cfg["data"]["root"]) / cfg["data"]["processed_subdir"]


@pytest.fixture(scope="module")
def df():
    p = processed / "splits.csv"
    assert p.exists(), "splits.csv missing - run scripts/make_splits.py first"
    return pd.read_csv(p)


def test_all_rows_assigned(df):
    assert df["split"].isin(["train", "val", "test"]).all()


def test_no_patient_leakage(df):
    spans = df.groupby("patient_id")["split"].nunique()
    leaked = spans[spans > 1]
    assert leaked.empty, f"{len(leaked)} patients span multiple splits: {list(leaked.index[:5])}"


def test_splits_nonempty_and_have_both_classes(df):
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        assert len(d) > 0, f"{s} split is empty"
        assert set(d["label"].unique()) == {0, 1}, f"{s} split missing a class"


def test_no_duplicate_images(df):
    assert df["filename"].is_unique, "duplicate filenames in split table"