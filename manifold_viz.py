"""
manifold_viz.py
===============
Renders the 'manifold of meaning' that a prompt licenses in an LLM's
response space.

Rather than tracing token-by-token trajectories, this script:

  1. Samples 200 diverse completions from qwen3:14b (temperature 1.0).
  2. Embeds each full completion with nomic-embed-text (768-D vectors).
  3. Projects all 201 points (prompt + 200 completions) to 3-D via MDS
     on pairwise Euclidean distances of the raw, un-normalized vectors.
  4. Fits a Gaussian KDE to the 200 endpoint positions in 3-D.
  5. Extracts the isosurface enclosing 70% of the probability mass via
     marching cubes — the manifold's translucent 'envelope'.
  6. Renders an interactive Plotly figure: envelope mesh + cluster-coloured
     scatter points + prompt origin marker.

Philosophical claim
-------------------
Every prompt selects a bounded region of the LLM's output distribution.
Embedded in a metric space, that region traces a structured shape — not a
random cloud but a 'manifold of meaning' with measurable geometry: volume,
surface area, and connectivity.  The number of disconnected lobes tells us
whether the prompt licenses one coherent semantic neighbourhood or several
distinct ones.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import requests
from scipy.stats import gaussian_kde
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.measure import label as sk_label
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotly.graph_objects as go

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
PROMPT         = "Describe a nice meadow at sunrise. A lot of nature, nice day."
N_COMPLETIONS  = 200
MAX_TOKENS     = 80
TEMPERATURE    = 1.0
N_CLUSTERS     = 6
MAX_WORKERS    = 8
OLLAMA_URL     = "http://localhost:11434"
GEN_MODEL      = "qwen3:14b"
EMBED_MODEL    = "nomic-embed-text"
EMBED_DIM      = 768

KDE_BANDWIDTH  = "scott"   # scipy gaussian_kde bw_method; try 0.3, 0.5, etc.
MASS_THRESHOLD = 0.70      # isosurface encloses this fraction of KDE mass
KDE_GRID_RES   = 60        # voxels per axis for KDE evaluation
MARGIN_FRAC    = 0.20      # bounding-box padding on each side

# Matplotlib static PNG: use every Nth face to keep rendering tractable
STATIC_MESH_DOWNSAMPLE = 6

COMPLETIONS_CACHE     = "completions.json"
ENDPOINTS_CACHE       = "endpoints_only.npy"
ENDPOINTS_INDEX_CACHE = "endpoints_only_index.json"

OUT_HTML = "meadow_manifold_envelope.html"
OUT_PNG  = "meadow_manifold_envelope_static.png"

# 6 visually distinct colours for clusters
CLUSTER_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A",
    "#457B9D", "#F4A261", "#6D6875",
]

# ===========================================================================
# 2. OLLAMA HELPERS
# ===========================================================================

def _post_with_retry(url: str, payload: dict, retries: int = 3,
                     timeout: int = 120) -> dict:
    """POST to Ollama with exponential-backoff retry on connection errors."""
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Ollama request failed after {retries} attempts: {exc}"
                ) from exc
            print(f"\n  [retry {attempt+1}/{retries}] {exc} — waiting {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
        except requests.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc


def generate_one(prompt: str, idx: int) -> str:
    """Generate one completion from Ollama with qwen3 thinking disabled."""
    payload = {
        "model":   GEN_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "think":   False,          # top-level field; suppresses qwen3 CoT
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    data = _post_with_retry(f"{OLLAMA_URL}/api/generate", payload)
    raw  = data.get("response", "").strip()
    # Defensive: strip any <think>…</think> block that still slips through
    raw  = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def embed_one(text: str) -> np.ndarray:
    """Return a raw 768-D embedding vector for *text*."""
    payload = {"model": EMBED_MODEL, "prompt": text}
    data    = _post_with_retry(f"{OLLAMA_URL}/api/embeddings", payload)
    return np.array(data["embedding"], dtype=np.float32)


# ===========================================================================
# 3. GENERATE COMPLETIONS  (parallel, cached)
# ===========================================================================

def _validate_completions(completions: list[str]) -> None:
    """Abort before writing cache if too many completions are empty/short."""
    min_good  = int(N_COMPLETIONS * 0.90)
    min_chars = 30
    good = [c for c in completions if c and len(c.strip()) >= min_chars]
    if len(good) < min_good:
        print(f"\nERROR: only {len(good)}/{len(completions)} completions have "
              f">= {min_chars} chars (need >= {min_good}).")
        print("First 5 raw responses:")
        for i, c in enumerate(completions[:5]):
            print(f"  [{i}] ({len(c)} chars) {repr(c)}")
        raise RuntimeError(
            f"Completion quality check failed: {len(good)}/{len(completions)} usable. "
            "Delete completions.json and rerun — verify 'think': false is working."
        )


def load_or_generate_completions() -> list[str]:
    if os.path.exists(COMPLETIONS_CACHE):
        print(f"[stage 1] Loading completions from cache: {COMPLETIONS_CACHE}")
        with open(COMPLETIONS_CACHE) as f:
            completions = json.load(f)
        print(f"          Loaded {len(completions)} completions.")
        return completions

    print(f"[stage 1] Generating {N_COMPLETIONS} completions with {GEN_MODEL} …")
    completions: list[str] = [None] * N_COMPLETIONS

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, PROMPT, i): i
                   for i in range(N_COMPLETIONS)}
        with tqdm(total=N_COMPLETIONS, desc="  generating") as pbar:
            for fut in as_completed(futures):
                completions[futures[fut]] = fut.result()
                pbar.update(1)

    _validate_completions(completions)

    print("\n  First 3 completions (verify real prose before continuing):")
    for i in range(min(3, len(completions))):
        print(f"  [{i}] {completions[i]}\n")

    with open(COMPLETIONS_CACHE, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"          Saved to {COMPLETIONS_CACHE}.")
    return completions


# ===========================================================================
# 4. EMBED ENDPOINTS ONLY  (prompt + full completions, parallel, cached)
# ===========================================================================

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_or_embed_endpoints(
    completions: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed the prompt (index 0) and all full completions (indices 1..N).
    No partial snapshots — one vector per response.

    Returns:
        prompt_emb  – (768,)            raw embedding of the prompt
        comp_embs   – (N_COMPLETIONS, 768)  raw embeddings of completions
    """
    texts = [PROMPT] + completions          # 201 strings total

    if os.path.exists(ENDPOINTS_CACHE) and os.path.exists(ENDPOINTS_INDEX_CACHE):
        print(f"[stage 2] Loading embedding cache from {ENDPOINTS_CACHE} …")
        cache_vecs = np.load(ENDPOINTS_CACHE)
        with open(ENDPOINTS_INDEX_CACHE) as f:
            cache_index: dict[str, int] = json.load(f)
    else:
        cache_vecs  = np.zeros((0, EMBED_DIM), dtype=np.float32)
        cache_index = {}

    hashes  = [_content_hash(t) for t in texts]
    missing = [i for i, h in enumerate(hashes) if h not in cache_index]

    if missing:
        print(f"          Embedding {len(missing)} new texts with {EMBED_MODEL} …")
        new_vecs: dict[int, np.ndarray] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(embed_one, texts[i]): i for i in missing}
            with tqdm(total=len(missing), desc="  embedding") as pbar:
                for fut in as_completed(futures):
                    new_vecs[futures[fut]] = fut.result()
                    pbar.update(1)

        next_row = len(cache_index)
        rows = [cache_vecs] if cache_vecs.shape[0] > 0 else []
        for i in missing:
            cache_index[hashes[i]] = next_row
            rows.append(new_vecs[i].reshape(1, -1))
            next_row += 1
        cache_vecs = np.vstack(rows)
        np.save(ENDPOINTS_CACHE, cache_vecs)
        with open(ENDPOINTS_INDEX_CACHE, "w") as f:
            json.dump(cache_index, f)
        print(f"          Cache updated ({cache_vecs.shape[0]} rows).")
    else:
        print(f"          All {len(texts)} texts found in cache — no API calls needed.")

    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, h in enumerate(hashes):
        out[i] = cache_vecs[cache_index[h]]

    return out[0], out[1:]          # prompt_emb, comp_embs


# ===========================================================================
# 5. PROJECT TO 3-D  (MDS on Euclidean distances, raw un-normalized vectors)
# ===========================================================================

def project_to_3d(
    prompt_emb: np.ndarray,
    comp_embs:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        prompt_3d    – (3,)              MDS coordinate of the prompt
        endpoints_3d – (N_COMPLETIONS, 3) MDS coordinates of completions
    """
    all_embs = np.vstack([prompt_emb.reshape(1, -1), comp_embs])   # (201, 768)
    n = all_embs.shape[0]

    print(f"[stage 3] Computing {n}×{n} Euclidean distance matrix …")
    dist = euclidean_distances(all_embs).astype(np.float64)
    dist = (dist + dist.T) / 2          # enforce exact symmetry
    np.fill_diagonal(dist, 0.0)

    print("[stage 3] Running MDS (this may take a minute) …")
    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        n_init=4,
        max_iter=400,
        random_state=42,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(dist).astype(np.float32)
    print(f"          MDS stress: {mds.stress_:.6f}")

    # Geometry diagnostic on the raw 768-D distance matrix
    d_to_prompt = dist[0, 1:]
    ep_pairs    = dist[np.ix_(range(1, n), range(1, n))]
    upper       = ep_pairs[np.triu_indices(n - 1, k=1)]
    cv = d_to_prompt.std() / d_to_prompt.mean() if d_to_prompt.mean() > 0 else 0

    print()
    print("  ── Geometry diagnostic (768-D Euclidean distances) ─────────────")
    print(f"  Prompt → endpoints    mean={d_to_prompt.mean():.4f}  "
          f"std={d_to_prompt.std():.4f}  "
          f"min={d_to_prompt.min():.4f}  max={d_to_prompt.max():.4f}")
    print(f"  Endpoint ↔ endpoint   mean={upper.mean():.4f}  "
          f"std={upper.std():.4f}  "
          f"min={upper.min():.4f}  max={upper.max():.4f}")
    print(f"  CoV of prompt→endpoint distances: {cv:.4f}  "
          f"({'non-spherical' if cv > 0.05 else 'near-spherical'})")
    print()

    return coords[0], coords[1:]        # prompt_3d, endpoints_3d


# ===========================================================================
# 6. CLUSTER IN 768-D SPACE  (more reliable than clustering after MDS)
# ===========================================================================

def cluster_in_highdim(
    comp_embs:   np.ndarray,
    completions: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    KMeans on raw 768-D embeddings.

    Returns:
        labels      – (N_COMPLETIONS,) int32 cluster indices
        reps        – full representative completion text per cluster
        reps_short  – 60-char truncations for the legend
    """
    print(f"[stage 4] KMeans (k={N_CLUSTERS}) in 768-D embedding space …")
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(comp_embs).astype(np.int32)

    reps:       list[str] = []
    reps_short: list[str] = []

    print()
    print("━" * 65)
    print(" CLUSTER REPRESENTATIVES")
    print("━" * 65)
    for k in range(N_CLUSTERS):
        mask = np.where(labels == k)[0]
        if len(mask) == 0:
            reps.append("")
            reps_short.append(f"Cluster {k} (empty)")
            print(f"  cluster {k}: 0 members — WARNING: no representative")
            continue
        centroid = km.cluster_centers_[k]
        dists    = np.linalg.norm(comp_embs[mask] - centroid, axis=1)
        rep_i    = int(mask[np.argmin(dists)])
        rep_text = completions[rep_i]
        short    = (rep_text[:57] + "…") if len(rep_text) > 60 else rep_text
        reps.append(rep_text)
        reps_short.append(short)
        warn = "  ⚠ WARNING: short/empty" if len(rep_text.strip()) < 10 else ""
        print(f"  cluster {k}: {len(mask):3d} members  |  "
              f"rep [{rep_i}] ({len(rep_text)} chars){warn}")
        print(f"    {repr(rep_text)}")
    print("━" * 65)
    print()

    return labels, reps, reps_short


# ===========================================================================
# 7. KDE + ISOSURFACE
# ===========================================================================

def compute_kde_isosurface(endpoints_3d: np.ndarray) -> dict:
    """
    Fit a Gaussian KDE to the 3-D endpoint cloud, evaluate on a regular
    grid, and extract the MASS_THRESHOLD isosurface via marching cubes.

    Returns a dict with keys:
        verts_world  – (V, 3) isosurface vertices in world coordinates
        faces        – (F, 3) triangle face indices
        density_grid – (R, R, R) KDE values on the grid
        threshold    – density level at the isosurface
        spacing      – (dx, dy, dz) voxel dimensions
        lo           – (3,) world-space grid origin
    """
    print(f"[stage 5] Fitting Gaussian KDE (bw_method={KDE_BANDWIDTH!r}) …")
    kde = gaussian_kde(endpoints_3d.T, bw_method=KDE_BANDWIDTH)

    lo = endpoints_3d.min(axis=0)
    hi = endpoints_3d.max(axis=0)
    margin = (hi - lo) * MARGIN_FRAC
    lo -= margin
    hi += margin

    print(f"[stage 5] Evaluating KDE on {KDE_GRID_RES}³ grid …")
    axes   = [np.linspace(lo[d], hi[d], KDE_GRID_RES) for d in range(3)]
    xi, yi, zi = np.meshgrid(*axes, indexing="ij")
    grid_pts   = np.vstack([xi.ravel(), yi.ravel(), zi.ravel()])
    density    = kde(grid_pts).reshape(KDE_GRID_RES, KDE_GRID_RES, KDE_GRID_RES)

    # Density threshold that encloses MASS_THRESHOLD of total probability mass
    flat_desc = np.sort(density.ravel())[::-1]
    cumsum    = np.cumsum(flat_desc) / flat_desc.sum()
    idx       = int(np.searchsorted(cumsum, MASS_THRESHOLD))
    threshold = float(flat_desc[min(idx, len(flat_desc) - 1)])
    print(f"          {MASS_THRESHOLD*100:.0f}% mass threshold: {threshold:.4e}")

    spacing = tuple((hi[d] - lo[d]) / (KDE_GRID_RES - 1) for d in range(3))

    print("[stage 5] Extracting isosurface with marching cubes …")
    verts, faces, _normals, _ = marching_cubes(
        density, level=threshold, spacing=spacing
    )
    # marching_cubes with spacing returns verts in [0, span] space; shift to world
    verts_world = verts + lo

    print(f"          Isosurface: {len(verts_world):,} vertices, {len(faces):,} faces.")
    return dict(
        verts_world  = verts_world,
        faces        = faces,
        density_grid = density,
        threshold    = threshold,
        spacing      = spacing,
        lo           = lo,
    )


# ===========================================================================
# 8. DIAGNOSTICS
# ===========================================================================

def print_diagnostics(endpoints_3d: np.ndarray, iso: dict) -> None:
    print()
    print("━" * 65)
    print(" MANIFOLD DIAGNOSTICS")
    print("━" * 65)

    lo_pts = endpoints_3d.min(axis=0)
    hi_pts = endpoints_3d.max(axis=0)
    print("  Endpoint cloud bounding box (3-D MDS coordinates):")
    for axis, label in enumerate("xyz"):
        span = hi_pts[axis] - lo_pts[axis]
        print(f"    {label}: [{lo_pts[axis]:.4f}, {hi_pts[axis]:.4f}]"
              f"  span = {span:.4f}")

    dx, dy, dz  = iso["spacing"]
    cell_vol    = dx * dy * dz
    binary      = iso["density_grid"] >= iso["threshold"]
    volume      = float(np.sum(binary)) * cell_vol
    area        = mesh_surface_area(iso["verts_world"], iso["faces"])
    n_comp      = int(sk_label(binary).max())

    print(f"\n  Isosurface (encloses {MASS_THRESHOLD*100:.0f}% of KDE mass):")
    print(f"    Volume         : {volume:.6f}  (MDS units)³")
    print(f"    Surface area   : {area:.6f}  (MDS units)²")
    print(f"    Connected lobes: {n_comp}")
    if n_comp == 1:
        print("    → Single connected blob — one coherent semantic neighbourhood.")
    else:
        print(f"    → {n_comp} disconnected lobes — the prompt licenses "
              f"{n_comp} distinct semantic regions.")

    print("━" * 65)
    print()


# ===========================================================================
# 9. PLOTLY INTERACTIVE FIGURE
# ===========================================================================

def build_plotly_figure(
    endpoints_3d: np.ndarray,
    prompt_3d:    np.ndarray,
    completions:  list[str],
    labels:       np.ndarray,
    reps_short:   list[str],
    iso:          dict,
) -> go.Figure:
    print("[stage 6] Building Plotly figure …")

    verts = iso["verts_world"]
    faces = iso["faces"]
    fig   = go.Figure()

    # Translucent isosurface envelope
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity    = 0.25,
        color      = "lightgray",
        flatshading= False,
        lighting   = dict(diffuse=0.9, specular=0.2, roughness=0.5, fresnel=0.2),
        lightposition = dict(x=100, y=200, z=150),
        name       = f"Envelope ({int(MASS_THRESHOLD*100)}% mass)",
        showlegend = True,
        hoverinfo  = "skip",
    ))

    # One scatter trace per cluster (cleaner than one trace per point)
    for k in range(N_CLUSTERS):
        mask  = np.where(labels == k)[0]
        if len(mask) == 0:
            continue
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        pts   = endpoints_3d[mask]
        hover = [completions[i] for i in mask]
        label = f"Cluster {k}: {reps_short[k]}" if reps_short[k] else f"Cluster {k}"

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode       = "markers",
            marker     = dict(size=5, color=color,
                              line=dict(color="white", width=0.4)),
            name       = label,
            legendgroup= f"cluster_{k}",
            showlegend = True,
            hovertext  = hover,
            hoverinfo  = "text",
        ))

    # Prompt origin — large black diamond with gold outline
    fig.add_trace(go.Scatter3d(
        x=[prompt_3d[0]], y=[prompt_3d[1]], z=[prompt_3d[2]],
        mode    = "markers",
        marker  = dict(size=14, symbol="diamond", color="black",
                       line=dict(color="gold", width=3)),
        name    = "Prompt origin",
        hovertext = PROMPT,
        hoverinfo = "text",
    ))

    subtitle2 = (
        f"{N_COMPLETIONS} completions, embedded by {EMBED_MODEL}, "
        f"projected to 3D (MDS), "
        f"envelope = {int(MASS_THRESHOLD*100)}% mass isosurface"
    )
    fig.update_layout(
        title=dict(
            text=(
                "Manifold of meaning licensed by one prompt<br>"
                f"<sup>{PROMPT}</sup><br>"
                f"<sup><i>{subtitle2}</i></sup>"
            ),
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title="MDS-1",
            yaxis_title="MDS-2",
            zaxis_title="MDS-3",
        ),
        legend=dict(
            title="Clusters (representative completion)",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, b=0, t=100),
    )
    return fig


# ===========================================================================
# 10. STATIC MATPLOTLIB 4-VIEW PNG
# ===========================================================================

def build_static_figure(
    endpoints_3d: np.ndarray,
    prompt_3d:    np.ndarray,
    labels:       np.ndarray,
    reps_short:   list[str],
    iso:          dict,
) -> None:
    print("[stage 7] Building static 4-view PNG …")

    verts = iso["verts_world"]
    faces = iso["faces"]

    # Subsample faces so Poly3DCollection stays tractable
    faces_ds  = faces[::STATIC_MESH_DOWNSAMPLE]
    tri_verts = verts[faces_ds]          # (M, 3, 3) — each row is one triangle

    view_angles = [
        (25,  45,  "Front-left"),
        (25, 135,  "Front-right"),
        (60,  45,  "Top"),
        (10,  10,  "Side"),
    ]

    # Axis limits from the point cloud (Poly3DCollection doesn't auto-expand)
    all_pts = np.vstack([endpoints_3d, prompt_3d.reshape(1, -1)])
    pad     = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.08
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    zlim = (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)

    fig = plt.figure(figsize=(14, 11))
    legend_handles: list = []
    legend_labels:  list[str] = []

    for ax_i, (elev, azim, view_name) in enumerate(view_angles):
        ax = fig.add_subplot(2, 2, ax_i + 1, projection="3d")
        ax.set_title(view_name, fontsize=9, pad=4)
        ax.set_xlabel("MDS-1", fontsize=7, labelpad=2)
        ax.set_ylabel("MDS-2", fontsize=7, labelpad=2)
        ax.set_zlabel("MDS-3", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        # Isosurface mesh
        poly = Poly3DCollection(
            tri_verts, alpha=0.12,
            facecolor="lightgray", edgecolor="none",
        )
        ax.add_collection3d(poly)

        # Cluster scatter points
        legend_added: set = set()
        for k in range(N_CLUSTERS):
            mask  = np.where(labels == k)[0]
            if len(mask) == 0:
                continue
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            pts   = endpoints_3d[mask]
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, s=14, zorder=5,
                edgecolors="white", linewidths=0.3,
            )
            if ax_i == 0 and k not in legend_added:
                legend_handles.append(sc)
                legend_labels.append(f"C{k}: {reps_short[k][:40]}")
                legend_added.add(k)

        # Prompt origin star
        star = ax.scatter(
            *prompt_3d, color="black", s=150, marker="*", zorder=10,
            edgecolors="gold", linewidths=0.9,
        )
        if ax_i == 0:
            legend_handles.append(star)
            legend_labels.append("Prompt origin")

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=4,
        fontsize=7, framealpha=0.85,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f'Manifold of meaning licensed by one prompt\n"{PROMPT}"',
        fontsize=10, y=0.99,
    )
    plt.tight_layout(rect=[0, 0.09, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"          Saved {OUT_PNG}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print(" LLM RESPONSE MANIFOLD VISUALISER  (envelope edition)")
    print("=" * 65)
    print(f" Prompt          : {PROMPT}")
    print(f" Model           : {GEN_MODEL}  |  Embed: {EMBED_MODEL}")
    print(f" Completions     : {N_COMPLETIONS}  |  Clusters: {N_CLUSTERS}")
    print(f" KDE bandwidth   : {KDE_BANDWIDTH}  |  Mass threshold: {MASS_THRESHOLD}")
    print(f" Grid resolution : {KDE_GRID_RES}³")
    print("=" * 65)

    completions               = load_or_generate_completions()
    prompt_emb, comp_embs     = load_or_embed_endpoints(completions)
    prompt_3d,  endpoints_3d  = project_to_3d(prompt_emb, comp_embs)
    labels, reps, reps_short  = cluster_in_highdim(comp_embs, completions)
    iso                       = compute_kde_isosurface(endpoints_3d)
    print_diagnostics(endpoints_3d, iso)

    fig = build_plotly_figure(
        endpoints_3d, prompt_3d, completions, labels, reps_short, iso
    )
    fig.write_html(OUT_HTML, include_plotlyjs=True)
    print(f"[stage 6] Saved interactive figure → {OUT_HTML}")

    build_static_figure(endpoints_3d, prompt_3d, labels, reps_short, iso)
    print(f"[stage 7] Saved static figure       → {OUT_PNG}")

    print()
    print("Done.  Open", OUT_HTML, "in a browser to explore the manifold.")


if __name__ == "__main__":
    main()
