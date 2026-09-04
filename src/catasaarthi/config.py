"""Load the single source of truth: configs/config.yaml.

find_project_root() walks up from the current working directory until it
finds configs/config.yaml, so scripts, tests and notebooks all load the
SAME config no matter where they run from.

Two environment variables can relocate the data paths WITHOUT editing the
committed config - used when running on Kaggle/Colab:
  CATASAARTHI_WORK_ROOT : where we WRITE (processed/, models/, outputs/)
  CATASAARTHI_RAW_ROOT  : where the dataset is READ from (full_df.csv + images)
Locally neither is set, so paths come straight from config.yaml.
"""
import os
from pathlib import Path
import yaml


def find_project_root(start=None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "configs" / "config.yaml").exists():
            return cand
    raise FileNotFoundError(
        f"configs/config.yaml not found when searching up from {p}. "
        "Run from inside the project (D:\\CataSaarthi)."
    )


def load_config(path=None) -> dict:
    root = find_project_root()
    cfg_path = Path(path) if path else root / "configs" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["_root"] = str(root)

    data = cfg["data"]
    data["root"] = os.environ.get("CATASAARTHI_WORK_ROOT", data["root"])
    data["raw_root"] = os.environ.get(
        "CATASAARTHI_RAW_ROOT",
        str(Path(data["root"]) / data["raw_subdir"]),
    )
    return cfg