#!/usr/bin/env python3
"""Generate an interactive static dashboard for project B in dashboard/index.html."""

import base64
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import umap
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DASH = ROOT / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)

SAMPLE = "NCBI10"


def img_to_dataurl(path: Path) -> str:
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def main():
    # Load data
    hopt0 = np.load(OUT / f"{SAMPLE}_h_optimus_0_embeddings.npy")
    uni2 = np.load(OUT / f"{SAMPLE}_uni2_h_embeddings.npy")
    spots = pd.read_csv(OUT / f"{SAMPLE}_h_optimus_0_spots.csv")
    with open(OUT / f"{SAMPLE}_embeddings_summary.json") as f:
        summary = json.load(f)
    with open(OUT / f"{SAMPLE}_summary.json") as f:
        wsi_summary = json.load(f)

    adata = sc.read_h5ad(ROOT / "data" / "st" / f"{SAMPLE}.h5ad")
    total_counts = adata.obs["total_counts"].values.astype(float)
    n_genes = adata.obs["n_genes_by_counts"].values.astype(float)

    # UMAP projections
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    umap_hopt0 = reducer.fit_transform(hopt0).tolist()
    umap_uni2 = reducer.fit_transform(uni2).tolist()

    # Per-spot similarity
    sims = np.diag(cosine_similarity(hopt0, uni2))

    # Thumbnail
    thumb_url = img_to_dataurl(OUT / f"{SAMPLE}_thumbnail.png")

    # Image scale: WSI -> thumbnail
    img_h = wsi_summary["wsi"]["dimensions"][1]
    thumb_path = OUT / f"{SAMPLE}_thumbnail.png"
    from PIL import Image
    thumb_w, thumb_h = Image.open(thumb_path).size
    scale = img_h / thumb_h

    points = [
        {
            "id": row.spot_id,
            "x": float(row.x) / scale,
            "y": thumb_h - float(row.y) / scale,
            "total_counts": float(total_counts[i]),
            "n_genes": float(n_genes[i]),
            "hopt0_x": umap_hopt0[i][0],
            "hopt0_y": umap_hopt0[i][1],
            "uni2_x": umap_uni2[i][0],
            "uni2_y": umap_uni2[i][1],
            "sim": float(sims[i]),
        }
        for i, row in enumerate(spots.itertuples())
    ]

    data = {
        "title": "Pathology Embeddings Quick Tour",
        "sample": SAMPLE,
        "stats": {
            "wsi_dims": wsi_summary["wsi"]["dimensions"],
            "mpp": round(float(wsi_summary["wsi"]["mpp"]), 3),
            "n_spots": int(summary["n_spots"]),
            "n_genes": int(adata.n_vars),
            "embedding_dim": int(summary["embedding_dim"]),
            "mean_sim": round(float(summary["mean_cosine_similarity_hopt0_uni2"]), 4),
        },
        "thumb": thumb_url,
        "thumb_size": {"width": thumb_w, "height": thumb_h},
        "points": points,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{data['title']}</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{background:#0f172a; color:#e2e8f0;}}
  .card {{background:#1e293b; border-radius:1rem; padding:1.5rem;}}
</style>
</head>
<body class="font-sans">
<div class="max-w-7xl mx-auto p-6">
  <h1 class="text-3xl font-bold mb-2">{data['title']}</h1>
  <p class="text-slate-400 mb-6">HEST-1k sample <span class="font-mono">{data['sample']}</span> — H-Optimus-0 vs UNI2-h tile embeddings</p>

  <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
    <div class="card"><div class="text-sm text-slate-400">WSI dimensions</div><div class="text-xl font-semibold">{data['stats']['wsi_dims'][0]} × {data['stats']['wsi_dims'][1]}</div></div>
    <div class="card"><div class="text-sm text-slate-400">MPP</div><div class="text-xl font-semibold">{data['stats']['mpp']}</div></div>
    <div class="card"><div class="text-sm text-slate-400">Spots</div><div class="text-xl font-semibold">{data['stats']['n_spots']}</div></div>
    <div class="card"><div class="text-sm text-slate-400">Genes</div><div class="text-xl font-semibold">{data['stats']['n_genes']}</div></div>
    <div class="card"><div class="text-sm text-slate-400">Embedding dim</div><div class="text-xl font-semibold">{data['stats']['embedding_dim']}</div></div>
    <div class="card"><div class="text-sm text-slate-400">Mean cosine sim</div><div class="text-xl font-semibold">{data['stats']['mean_sim']}</div></div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div class="card"><div id="wsi-plot" class="h-96"></div></div>
    <div class="card"><div id="hopt0-umap" class="h-96"></div></div>
    <div class="card"><div id="uni2-umap" class="h-96"></div></div>
    <div class="card"><div id="sim-hist" class="h-96"></div></div>
  </div>
</div>

<script>
const data = {json.dumps(data, indent=2)};

const color = data.points.map(p => p.total_counts);

function scatterTrace(x, y, color, title, hover) {{
  return {{
    type: 'scatter',
    mode: 'markers',
    x: x,
    y: y,
    marker: {{
      color: color,
      colorscale: 'Viridis',
      colorbar: {{title: 'Total UMI counts', thickness: 12}},
      size: 8,
      line: {{width: 0.5, color: '#0f172a'}}
    }},
    text: hover,
    hoverinfo: 'text'
  }};
}}

const layout = (title, img) => {{
  const l = {{
    title: {{text: title, font: {{color: '#e2e8f0'}}}},
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{color: '#94a3b8'}},
    margin: {{t: 40, r: 40, b: 40, l: 50}}
  }};
  if (img) {{
    l.images = [{{
      source: data.thumb,
      x: 0,
      y: data.thumb_size.height,
      sizex: data.thumb_size.width,
      sizey: data.thumb_size.height,
      xref: 'x',
      yref: 'y',
      xanchor: 'left',
      yanchor: 'top',
      sizing: 'stretch',
      layer: 'below'
    }}];
    l.xaxis = {{range: [0, data.thumb_size.width], scaleanchor: 'y'}};
    l.yaxis = {{range: [data.thumb_size.height, 0]}};
  }}
  return l;
}};

Plotly.newPlot('wsi-plot', [scatterTrace(
  data.points.map(p => p.x),
  data.points.map(p => p.y),
  color,
  'Spots on H&E thumbnail',
  data.points.map(p => `Spot ${{p.id}}<br>UMIs: ${{p.total_counts.toFixed(0)}}<br>Genes: ${{p.n_genes.toFixed(0)}}`)
)], layout('Spots on H&E thumbnail', true));

Plotly.newPlot('hopt0-umap', [scatterTrace(
  data.points.map(p => p.hopt0_x),
  data.points.map(p => p.hopt0_y),
  color,
  'H-Optimus-0 UMAP',
  data.points.map(p => `Spot ${{p.id}}<br>Sim: ${{p.sim.toFixed(3)}}`)
)], layout('H-Optimus-0 UMAP', false));

Plotly.newPlot('uni2-umap', [scatterTrace(
  data.points.map(p => p.uni2_x),
  data.points.map(p => p.uni2_y),
  color,
  'UNI2-h UMAP',
  data.points.map(p => `Spot ${{p.id}}<br>Sim: ${{p.sim.toFixed(3)}}`)
)], layout('UNI2-h UMAP', false));

Plotly.newPlot('sim-hist', [{{
  type: 'histogram',
  x: data.points.map(p => p.sim),
  marker: {{color: '#38bdf8'}},
  nbinsx: 20
}}], {{
  title: {{text: 'H-Optimus-0 vs UNI2-h cosine similarity per spot', font: {{color: '#e2e8f0'}}}},
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: {{color: '#94a3b8'}},
  xaxis: {{title: 'cosine similarity'}},
  yaxis: {{title: 'count'}}
}});
</script>
</body>
</html>
"""
    (DASH / "index.html").write_text(html)
    print("Dashboard written to", DASH / "index.html")


if __name__ == "__main__":
    main()
