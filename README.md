# CataSaarthi AI

AI-assisted screening support for cataract - final-year B.E. (AI & Data Science) project, MMCOE Pune (SPPU), Group BE_P06.

NOTE: Not a diagnostic tool. CataSaarthi produces a "suspected cataract" / "normal" screening signal on retinal fundus images to support, never replace, professional eye examination. Every output requires evaluation by a qualified clinician.

## What it does
A fundus image is checked for quality, then a deep-learning model produces a calibrated Normal vs Suspected-Cataract prediction with an uncertainty estimate. Explainable-AI overlays (Grad-CAM / Score-CAM) show which regions drove the prediction, and a bounded assistant produces a structured, non-diagnostic summary. A human reviewer stays in the loop.

## Approach (phased)
Phase 1 (current): a Streamlit app on top of a trustworthy prediction pipeline - data audit, baseline + transfer-learning models, calibration, image-quality gate, XAI.
Phase 2 (later, only if a real need appears): richer UI / API / database. We deliberately avoid over-engineering (no Docker / microservices / cloud) unless justified.

## Project structure
  app/                 Streamlit UI (Phase 1)
  configs/config.yaml  all paths, seed, model choices
  data/                local scratch only - real datasets live on D:\catasaarthi_data
  docs/                planning docs, diagrams
  notebooks/           exploration
  scripts/             runnable scripts (train, evaluate, ...)
  src/catasaarthi/     the importable package (data, models, xai, agent, inference)
  tests/               pytest tests (incl. safety + leakage checks)

## Setup
  py -3.13 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt

PyTorch here is the CPU build; model training runs on free cloud GPUs (Kaggle / Colab).

## Data
Datasets are NOT stored in this repo. Point data.root in configs/config.yaml at your local dataset folder. Planned sources: ODIR-5K (fundus, screening) and EyeQ (image-quality gate).

## Integrity and safety
No fabricated metrics, datasets, or citations. Patient-level train/val/test splits; the test set is locked and never used for model selection. Deterministic safety rules always override any generated text.

## Team
Aditi Dalvi, Srushti Deokar, Payal Gadge, Tanvi Shinde. Guide: Mrs. Swati Jakkan.
