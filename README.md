# Pathology Embeddings Quickstart

A minimal project for pathology WSI + spatial transcriptomics (HEST-1k) using two competing foundation models:
- **H-Optimus-0** (Bioptimus)
- **UNI2-h** (Mahmood Lab / Harvard)

It loads a small sample from HEST-1k, extracts 224×224 H&E tiles at 0.5 MPP, runs each model to get 1536-dimensional embeddings, and computes basic correlations with spatial gene expression.

## Data

- HEST-1k sample `NCBI10`:
  - `data/NCBI10.tif` — H&E whole-slide image (~30 MB)
  - `data/NCBI10.h5ad` — spatial transcriptomics expression matrix (~22 MB)

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN="hf_..."  # needs access to H-Optimus-0, UNI2-h, and MahmoodLab/hest
```

## Run

```bash
python scripts/download_hest_sample.py
python scripts/inspect_hest.py
python scripts/extract_embeddings.py
```

Outputs are written to `outputs/`:
- JSON summaries of WSI and ST data
- CSV of tile coordinates and embeddings
- PNG thumbnails of sample tiles
- Correlation report between embeddings and gene expression

## License

MIT
