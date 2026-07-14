#!/usr/bin/env python3
"""Inspect HEST-1k WSI and spatial transcriptomics data."""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import tiffslide
import scanpy as sc
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE = "NCBI10"


def inspect_wsi(path: Path):
    slide = tiffslide.TiffSlide(path)
    dims = slide.dimensions
    levels = slide.level_dimensions
    mpp = slide.properties.get("tiffslide.mpp-x") or slide.properties.get("openslide.mpp-x")
    objective = slide.properties.get("tiffslide.objective-power") or slide.properties.get("openslide.objective-power")

    # Read a thumbnail
    thumb = slide.get_thumbnail((512, 512))
    thumb_path = OUT / f"{SAMPLE}_thumbnail.png"
    thumb.save(thumb_path)

    summary = {
        "file": path.name,
        "dimensions": dims,
        "level_dimensions": levels,
        "mpp": mpp,
        "objective_power": objective,
        "thumbnail": str(thumb_path),
    }
    print("WSI summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def inspect_st(path: Path):
    adata = sc.read_h5ad(path)
    obs = adata.obs
    var = adata.var
    X = adata.X

    # Spatial coordinates are stored in adata.obsm['spatial'] for HEST-1k
    spatial = adata.obsm.get("spatial")
    if spatial is None:
        # fallback: look for any key containing spatial
        for k in adata.obsm.keys():
            if "spatial" in k.lower():
                spatial = adata.obsm[k]
                break

    summary = {
        "file": path.name,
        "n_spots": adata.n_obs,
        "n_genes": adata.n_vars,
        "obs_columns": list(obs.columns),
        "var_columns": list(var.columns),
        "spatial_key": "spatial" if "spatial" in adata.obsm else None,
        "spatial_shape": list(spatial.shape) if spatial is not None else None,
        "spatial_range": {
            "x_min": float(spatial[:, 0].min()),
            "x_max": float(spatial[:, 0].max()),
            "y_min": float(spatial[:, 1].min()),
            "y_max": float(spatial[:, 1].max()),
        } if spatial is not None else None,
        "expression_mean": float(X.mean()) if not issparse(X) else float(X.mean()),
        "expression_nonzero": float(np.count_nonzero(X) / X.size) if not issparse(X) else float(X.nnz / X.size),
    }

    print("ST summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save most variable genes
    if "n_counts" in adata.obs.columns:
        print("  total_umi_counts:", float(adata.obs["n_counts"].sum()))

    return summary


def main():
    wsi_path = DATA / f"wsis/{SAMPLE}.tif"
    st_path = DATA / f"st/{SAMPLE}.h5ad"

    wsi_summary = inspect_wsi(wsi_path)
    st_summary = inspect_st(st_path)

    with open(OUT / f"{SAMPLE}_summary.json", "w") as f:
        json.dump({"wsi": wsi_summary, "st": st_summary}, f, indent=2, default=str)

    print("\nDone. Output saved to", OUT)


if __name__ == "__main__":
    main()
