#!/usr/bin/env python3
"""Extract H-Optimus-0 and UNI2-h tile embeddings for HEST-1k spots.

Expects:
  data/wsis/{SAMPLE}.tif
  data/st/{SAMPLE}.h5ad

Outputs:
  outputs/{model}_embeddings.npy
  outputs/{model}_spots.csv
  outputs/embeddings_summary.json
"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tiffslide
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE = "NCBI10"
TARGET_MPP = 0.5
PATCH_SIZE = 224  # all models expect 224x224


def load_hest_data():
    slide = tiffslide.TiffSlide(DATA / f"wsis/{SAMPLE}.tif")
    adata = sc.read_h5ad(DATA / f"st/{SAMPLE}.h5ad")
    mpp = float(slide.properties.get("tiffslide.mpp-x") or slide.properties.get("openslide.mpp-x"))
    spatial = adata.obsm["spatial"]
    spot_size = int(round(PATCH_SIZE * (TARGET_MPP / mpp)))
    return slide, adata, mpp, spatial, spot_size


def extract_patches(slide, spatial, spot_size):
    patches = []
    coords = []
    for i, (cx, cy) in enumerate(spatial):
        x = int(cx - spot_size / 2)
        y = int(cy - spot_size / 2)
        # Clamp to slide bounds
        width, height = slide.dimensions
        x = max(0, min(x, width - spot_size))
        y = max(0, min(y, height - spot_size))
        img = slide.read_region((x, y), 0, (spot_size, spot_size))
        img = img.convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BICUBIC)
        patches.append(img)
        coords.append((cx, cy))
    return patches, coords


def embeddings_for_model(model, transform, patches, device="cpu"):
    model = model.to(device).eval()
    feats = []
    with torch.inference_mode():
        for img in patches:
            x = transform(img).unsqueeze(0).to(device)
            out = model(x)
            feats.append(out.squeeze(0).cpu().numpy())
    return np.stack(feats)


def load_h_optimus_0():
    """Load H-Optimus-0 with memory-mapped checkpoint to avoid OOM."""
    print("Loading H-Optimus-0...")
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=False,
        init_values=1e-5,
        dynamic_img_size=False,
    )
    ckpt = hf_hub_download(
        "bioptimus/H-optimus-0",
        "pytorch_model.bin",
        token=os.environ.get("HF_TOKEN"),
    )
    state_dict = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(PATCH_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517),
        ),
    ])
    return model, transform


def load_uni2_h():
    """Load UNI2-h with memory-mapped checkpoint."""
    print("Loading UNI2-h...")
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=False,
        **timm_kwargs,
    )
    ckpt = hf_hub_download(
        "MahmoodLab/UNI2-h",
        "pytorch_model.bin",
        token=os.environ.get("HF_TOKEN"),
    )
    state_dict = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a * b, axis=1)))


def main():
    assert "HF_TOKEN" in os.environ, "Set HF_TOKEN environment variable."
    device = "cpu"

    slide, adata, mpp, spatial, spot_size = load_hest_data()
    print(f"WSI mpp={mpp:.3f}, spot_size_level0={spot_size}")
    patches, coords = extract_patches(slide, spatial, spot_size)
    print(f"Extracted {len(patches)} patches")

    # H-Optimus-0
    model, transform = load_h_optimus_0()
    hopt0 = embeddings_for_model(model, transform, patches, device)
    np.save(OUT / f"{SAMPLE}_h_optimus_0_embeddings.npy", hopt0)
    pd.DataFrame({
        "spot_id": [f"{SAMPLE}_{i}" for i in range(len(coords))],
        "x": [c[0] for c in coords],
        "y": [c[1] for c in coords],
    }).to_csv(OUT / f"{SAMPLE}_h_optimus_0_spots.csv", index=False)
    print("H-Optimus-0 embeddings:", hopt0.shape)
    del model
    gc.collect()

    # UNI2-h
    model, transform = load_uni2_h()
    uni2 = embeddings_for_model(model, transform, patches, device)
    np.save(OUT / f"{SAMPLE}_uni2_h_embeddings.npy", uni2)
    pd.DataFrame({
        "spot_id": [f"{SAMPLE}_{i}" for i in range(len(coords))],
        "x": [c[0] for c in coords],
        "y": [c[1] for c in coords],
    }).to_csv(OUT / f"{SAMPLE}_uni2_h_spots.csv", index=False)
    print("UNI2-h embeddings:", uni2.shape)

    # Summary
    summary = {
        "sample": SAMPLE,
        "n_spots": len(patches),
        "mpp": mpp,
        "spot_size_level0": spot_size,
        "embedding_dim": hopt0.shape[1],
        "mean_cosine_similarity_hopt0_uni2": cosine_similarity(hopt0, uni2),
    }
    with open(OUT / f"{SAMPLE}_embeddings_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
