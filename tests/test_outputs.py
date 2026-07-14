from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def test_embeddings_summary():
    summary = OUT / "NCBI10_embeddings_summary.json"
    assert summary.exists()
    import json
    data = json.loads(summary.read_text())
    assert data["sample"] == "NCBI10"
    assert data["n_spots"] == 138
    assert data["embedding_dim"] == 1536


def test_h_optimus_0_embeddings():
    path = OUT / "NCBI10_h_optimus_0_embeddings.npy"
    assert path.exists()
    arr = np.load(path)
    assert arr.shape == (138, 1536)


def test_uni2_h_embeddings():
    path = OUT / "NCBI10_uni2_h_embeddings.npy"
    assert path.exists()
    arr = np.load(path)
    assert arr.shape == (138, 1536)
