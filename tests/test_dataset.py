"""
Dataset + preprocessing tests (Task 11).
Run:  pytest -q
"""
from pathlib import Path
import pandas as pd
import torch
import pytest

from catasaarthi.config import load_config
from catasaarthi.dataset import FundusDataset, build_image_index


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def index(cfg):
    return build_image_index(cfg)


def _splits(cfg):
    processed = Path(cfg["data"]["root"]) / cfg["data"]["processed_subdir"]
    return pd.read_csv(processed / "splits.csv")


def test_all_images_resolve(cfg, index):
    df = _splits(cfg)
    missing = [f for f in df["filename"] if f not in index]
    assert not missing, f"{len(missing)} images missing from originals, e.g. {missing[:5]}"


def test_lengths_match_splits(cfg, index):
    df = _splits(cfg)
    for split in ["train", "val", "test"]:
        ds = FundusDataset(split, cfg=cfg, image_index=index)
        assert len(ds) == int((df["split"] == split).sum())


def test_sample_tensor_shape_and_label(cfg, index):
    size = int(cfg["image"]["size"])
    ds = FundusDataset("val", cfg=cfg, image_index=index)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, size, size)
    assert x.dtype == torch.float32
    assert y in (0, 1)