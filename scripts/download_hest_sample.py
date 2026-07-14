#!/usr/bin/env python3
"""Download a small HEST-1k sample (WSI + spatial transcriptomics)."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

REPO = "MahmoodLab/hest"
SAMPLE = "NCBI10"


def download_file(path: str):
    local = DATA / Path(path).name
    if local.exists():
        print(f"Already downloaded: {local}")
        return local
    print(f"Downloading {path} ...")
    fetched = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=path,
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(DATA),
        local_dir_use_symlinks=False,
    )
    print(f"  -> {fetched}")
    return Path(fetched)


def main():
    assert "HF_TOKEN" in os.environ, "Set HF_TOKEN env var to access gated HEST-1k files."
    download_file(f"wsis/{SAMPLE}.tif")
    download_file(f"st/{SAMPLE}.h5ad")
    print("Done.")


if __name__ == "__main__":
    main()
