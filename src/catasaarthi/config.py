"""Load the single source of truth: configs/config.yaml.

find_project_root() walks up from the current working directory until it
finds configs/config.yaml, so scripts, tests and notebooks all load the
SAME config no matter where they run from.
"""
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
    cfg["_root"] = str(root)  # handy for callers
    return cfg