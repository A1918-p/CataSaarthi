"""FundusDataset: reads splits.csv, locates each eye's ORIGINAL image, and
returns (image_tensor, label) with our preprocessing applied.

build_image_index() scans the originals ONCE (raw/ODIR-5K, recursively) and
maps filename -> full path, so we don't care whether an image sits in the
'Training Images' or 'Testing Images' subfolder.
"""
from pathlib import Path
import cv2
import pandas as pd
from torch.utils.data import Dataset

from .config import load_config
from .preprocess import crop_fundus, build_transforms

IMG_EXT = {".jpg", ".jpeg", ".png"}


def build_image_index(cfg: dict) -> dict:
    originals = (Path(cfg["data"]["raw_root"])
                 / cfg["data"]["originals_subdir"])
    index = {}
    for p in originals.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            index[p.name] = p
    return index


class FundusDataset(Dataset):
    def __init__(self, split, cfg=None, transform=None, image_index=None):
        assert split in {"train", "val", "test"}
        self.cfg = cfg or load_config()
        size = int(self.cfg["image"]["size"])

        processed = (Path(self.cfg["data"]["root"])
                     / self.cfg["data"]["processed_subdir"])
        df = pd.read_csv(processed / "splits.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.transform = (transform if transform is not None
                          else build_transforms(size, train=(split == "train")))
        self.index = (image_index if image_index is not None
                      else build_image_index(self.cfg))

        missing = [f for f in self.df["filename"] if f not in self.index]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images not found in originals, "
                f"e.g. {missing[:5]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(str(self.index[row["filename"]]))   # BGR uint8
        if img is None:
            raise FileNotFoundError(f"cv2 could not read {row['filename']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_fundus(img)
        img = self.transform(image=img)["image"]             # 3 x H x W float
        return img, int(row["label"])