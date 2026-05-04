"""
manifold_viz.py
===============
Visualises LLM response trajectories as paths on a learned manifold.

Philosophical claim
-------------------
A language model defines an implicit probability distribution over
completions conditioned on a prompt.  When you sample many completions
from the same prompt and embed each one token-by-token, the resulting
point-clouds do *not* fill embedding space uniformly: they concentrate on
a low-dimensional sub-manifold that is "selected" by the prompt.  Each
completion is a directed trajectory on that manifold — starting at the
shared prompt-origin and diverging toward semantically distinct regions.
Dimensionality reduction (MDS on Euclidean distances of raw embeddings)
lets us render these trajectories in 3-D and observe the manifold's
geometry without the spherical distortion introduced by L2 normalisation:
how tightly the paths bundle early on, where they branch, whether
magnitude differences carry signal, and whether the endpoints cluster into
coherent semantic attractors.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import requests
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

import plotly.graph_objects as go

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
PROMPT = "Describe a nice meadow at sunrise. A lot of nature, nice day."
N_COMPLETIONS   = 100
MAX_TOKENS      = 80
TEMPERATURE     = 1.0
N_SNAPSHOTS     = 12   # incremental partials per completion (excludes prompt origin)
N_CLUSTERS      = 8
MAX_WORKERS     = 8
OLLAMA_URL      = "http://localhost:11434"
GEN_MODEL       = "qwen3:14b"
EMBED_MODEL     = "nomic-embed-text"
EMBED_DIM       = 768

COMPLETIONS_CACHE = "completions.json"
EMBEDDINGS_CACHE  = "embeddings.npy"
EMBED_INDEX_CACHE = "embeddings_index.json"

OUT_HTML = "meadow_trajectories_3d_unnormalized.html"
OUT_PNG  = "meadow_trajectories_3d_unnormalized_static.png"

# Cluster colours (up to 8)
CLUSTER_COLORS = [
    "#E63946", "#2A9D8F", "#E9C46A", "#457B9D",
    "#F4A261", "#6D6875", "#52B788", "#D62828",
]

# ===========================================================================
# 2. OLLAMA HELPERS
# ===========================================================================

def _post_with_retry(url: str, payload: dict, retries: int = 3, timeout: int = 120) -> dict:
    """POST to Ollama with exponential-backoff retry on connection errors."""
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Ollama request failed after {retries} attempts: {exc}") from exc
            print(f"\n  [retry {attempt+1}/{retries}] {exc} — waiting {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
        except requests.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc


def generate_one(prompt: str, idx: int) -> str:
    """Generate a single completion from Ollama with thinking disabled."""
    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "think": False,          # disable qwen3 chain-of-thought so tokens go to the response
        },
    }
    data = _post_with_retry(f"{OLLAMA_URL}/api/generate", payload)
    raw = data.get("response", "").strip()
    # Belt-and-suspenders: strip any residual think blocks
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def embed_one(text: str) -> np.ndarray:
    """Return a 768-D embedding vector for *text*."""
    payload = {"model": EMBED_MODEL, "prompt": text}
    data = _post_with_retry(f"{OLLAMA_URL}/api/embeddings", payload)
    vec = np.array(data["embedding"], dtype=np.float32)
    return vec


# ===========================================================================
# 3. GENERATE COMPLETIONS (parallel, cached)
# ===========================================================================

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
        futures = {pool.submit(generate_one, PROMPT, i): i for i in range(N_COMPLETIONS)}
        with tqdm(total=N_COMPLETIONS, desc="  generating") as pbar:
            for fut in as_completed(futures):
                i = futures[fut]
                completions[i] = fut.result()
                pbar.update(1)

    with open(COMPLETIONS_CACHE, "w") as f:
        json.dump(completions, f, indent=2)
    print(f"          Saved to {COMPLETIONS_CACHE}.")
    return completions


# ===========================================================================
# 4. BUILD INCREMENTAL SNAPSHOTS
# ===========================================================================

def build_snapshots(completions: list[str]) -> tuple[list[str], list[int], list[int]]:
    """
    Returns:
        texts        – flat list of all snapshot strings to embed
        comp_ids     – which completion each snapshot belongs to (-1 = prompt origin)
        snap_indices – snapshot index within its completion (0 = prompt origin)
    """
    print("[stage 2] Building incremental snapshots …")
    texts: list[str]   = []
    comp_ids: list[int]  = []
    snap_indices: list[int] = []

    # Prompt origin (shared across all trajectories, embedded once)
    texts.append(PROMPT)
    comp_ids.append(-1)
    snap_indices.append(0)

    for c_idx, completion in enumerate(completions):
        words = completion.split()
        total_words = len(words)
        if total_words == 0:
            # Empty completion — pad with prompt
            for s in range(N_SNAPSHOTS):
                texts.append(PROMPT)
                comp_ids.append(c_idx)
                snap_indices.append(s + 1)
            continue

        # Evenly-spaced word counts: 1 word … all words
        snap_word_counts = np.linspace(1, total_words, N_SNAPSHOTS, dtype=int)
        snap_word_counts = np.clip(snap_word_counts, 1, total_words)

        for s, k in enumerate(snap_word_counts):
            prefix = " ".join(words[:k])
            texts.append(PROMPT + " " + prefix)
            comp_ids.append(c_idx)
            snap_indices.append(s + 1)

    print(f"          {len(texts)} total snapshot texts "
          f"(1 origin + {N_COMPLETIONS}×{N_SNAPSHOTS} partials).")
    return texts, comp_ids, snap_indices


# ===========================================================================
# 5. EMBED EVERYTHING (parallel, cached by content hash)
# ===========================================================================

def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_or_embed(texts: list[str]) -> np.ndarray:
    """Return (len(texts), EMBED_DIM) array of raw (un-normalized) embeddings."""

    # Load existing cache
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(EMBED_INDEX_CACHE):
        print(f"[stage 3] Loading embedding cache from {EMBEDDINGS_CACHE} …")
        cache_vecs  = np.load(EMBEDDINGS_CACHE)
        with open(EMBED_INDEX_CACHE) as f:
            cache_index: dict[str, int] = json.load(f)  # hash -> row in cache_vecs
    else:
        cache_vecs  = np.zeros((0, EMBED_DIM), dtype=np.float32)
        cache_index = {}

    hashes = [_content_hash(t) for t in texts]
    missing_indices = [i for i, h in enumerate(hashes) if h not in cache_index]

    if missing_indices:
        print(f"          Embedding {len(missing_indices)} new texts with {EMBED_MODEL} …")
        new_vecs: dict[int, np.ndarray] = {}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(embed_one, texts[i]): i for i in missing_indices}
            with tqdm(total=len(missing_indices), desc="  embedding") as pbar:
                for fut in as_completed(futures):
                    i = futures[fut]
                    new_vecs[i] = fut.result()
                    pbar.update(1)

        # Append new rows to cache
        next_row = len(cache_index)
        rows_to_stack = [cache_vecs] if cache_vecs.shape[0] > 0 else []
        for i in missing_indices:
            h = hashes[i]
            cache_index[h] = next_row
            rows_to_stack.append(new_vecs[i].reshape(1, -1))
            next_row += 1
        cache_vecs = np.vstack(rows_to_stack)

        np.save(EMBEDDINGS_CACHE, cache_vecs)
        with open(EMBED_INDEX_CACHE, "w") as f:
            json.dump(cache_index, f)
        print(f"          Cache updated ({cache_vecs.shape[0]} total rows).")
    else:
        print(f"          All {len(texts)} texts found in cache — no new API calls.")

    # Assemble output array in original order — raw vectors, no normalization
    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    for i, h in enumerate(hashes):
        out[i] = cache_vecs[cache_index[h]]
    return out


# ===========================================================================
# 6. PROJECT TO 3-D (MDS on Euclidean distance, raw un-normalized embeddings)
# ===========================================================================

def project_to_3d(
    embeddings: np.ndarray,
    comp_ids: list[int],
    snap_indices: list[int],
) -> np.ndarray:
    print("[stage 4] Computing pairwise Euclidean distance matrix …")
    dist = euclidean_distances(embeddings).astype(np.float64)
    dist = (dist + dist.T) / 2   # ensure perfect symmetry
    np.fill_diagonal(dist, 0.0)

    print("[stage 4] Running MDS (this may take a minute) …")
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

    # ---- Diagnostic: distance spread in the raw distance matrix -------------
    # Locate prompt origin and all endpoint indices
    origin_idx    = next(i for i, c in enumerate(comp_ids) if c == -1)
    endpoint_idxs = [i for i, (c, s) in enumerate(zip(comp_ids, snap_indices))
                     if c >= 0 and s == N_SNAPSHOTS]

    d_prompt_to_ep = dist[origin_idx, endpoint_idxs]
    ep_pairs       = dist[np.ix_(endpoint_idxs, endpoint_idxs)]
    upper          = ep_pairs[np.triu_indices(len(endpoint_idxs), k=1)]

    print()
    print("  ── Geometry diagnostic (Euclidean distances in embedding space) ──")
    print(f"  Prompt → endpoints   mean={d_prompt_to_ep.mean():.4f}  "
          f"std={d_prompt_to_ep.std():.4f}  "
          f"min={d_prompt_to_ep.min():.4f}  max={d_prompt_to_ep.max():.4f}")
    print(f"  Endpoint ↔ endpoint  mean={upper.mean():.4f}  "
          f"std={upper.std():.4f}  "
          f"min={upper.min():.4f}  max={upper.max():.4f}")
    cv = d_prompt_to_ep.std() / d_prompt_to_ep.mean() if d_prompt_to_ep.mean() > 0 else 0
    print(f"  CoV (std/mean) of prompt→endpoint distances: {cv:.4f}  "
          f"({'non-spherical' if cv > 0.05 else 'near-spherical'})")
    print()

    return coords


# ===========================================================================
# 7. CLUSTER ENDPOINTS + FIND REPRESENTATIVES
# ===========================================================================

def cluster_endpoints(
    coords: np.ndarray,
    comp_ids: list[int],
    snap_indices: list[int],
    completions: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Returns:
        labels           – cluster label per completion (len = N_COMPLETIONS)
        representatives  – one representative completion string per cluster
        rep_truncated    – 60-char truncations for legend
    """
    print("[stage 5] Clustering endpoints with KMeans …")

    # Collect endpoint coords (last snapshot of each completion)
    endpoint_coords = np.zeros((N_COMPLETIONS, 3), dtype=np.float32)
    for flat_i, (c_idx, s_idx) in enumerate(zip(comp_ids, snap_indices)):
        if c_idx >= 0 and s_idx == N_SNAPSHOTS:
            endpoint_coords[c_idx] = coords[flat_i]

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(endpoint_coords)

    representatives: list[str] = []
    rep_truncated: list[str]   = []
    for k in range(N_CLUSTERS):
        mask  = np.where(labels == k)[0]
        if len(mask) == 0:
            representatives.append("")
            rep_truncated.append(f"Cluster {k}")
            continue
        centroid = kmeans.cluster_centers_[k]
        dists    = np.linalg.norm(endpoint_coords[mask] - centroid, axis=1)
        rep_idx  = mask[np.argmin(dists)]
        rep_text = completions[rep_idx]
        representatives.append(rep_text)
        rep_truncated.append((rep_text[:57] + "…") if len(rep_text) > 60 else rep_text)
        print(f"  cluster {k}: {rep_truncated[-1]}")

    return labels, representatives, rep_truncated


# ===========================================================================
# 8. PLOTLY INTERACTIVE FIGURE
# ===========================================================================

def _smooth_trajectory(pts: np.ndarray, n_out: int = 120) -> np.ndarray:
    """Cubic B-spline through *pts* (shape N×3), returns (n_out, 3)."""
    n = len(pts)
    if n < 4:
        return pts
    t = np.linspace(0, 1, n)
    t_fine = np.linspace(0, 1, n_out)
    try:
        spl = make_interp_spline(t, pts, k=3)
        return spl(t_fine)
    except Exception:
        return pts


def build_plotly_figure(
    coords: np.ndarray,
    comp_ids: list[int],
    snap_indices: list[int],
    completions: list[str],
    labels: np.ndarray,
    rep_truncated: list[str],
) -> go.Figure:
    print("[stage 6] Building interactive Plotly figure …")

    # Map flat index → coords
    # Gather per-completion trajectory points
    traj_pts:  list[list[np.ndarray]] = [[] for _ in range(N_COMPLETIONS)]
    endpoint_flat_idx: list[int] = [-1] * N_COMPLETIONS
    origin_flat_idx = -1

    for flat_i, (c_idx, s_idx) in enumerate(zip(comp_ids, snap_indices)):
        if c_idx == -1:
            origin_flat_idx = flat_i
        else:
            traj_pts[c_idx].append((s_idx, coords[flat_i]))
            if s_idx == N_SNAPSHOTS:
                endpoint_flat_idx[c_idx] = flat_i

    # Sort each trajectory by snapshot index
    for c_idx in range(N_COMPLETIONS):
        traj_pts[c_idx].sort(key=lambda x: x[0])

    origin = coords[origin_flat_idx]

    fig = go.Figure()

    legend_shown = set()

    for c_idx in range(N_COMPLETIONS):
        k = int(labels[c_idx])
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        show_legend = k not in legend_shown
        legend_shown.add(k)

        # Build array: origin + trajectory points
        pts_raw = np.vstack(
            [origin] + [p for _, p in traj_pts[c_idx]]
        )
        pts_smooth = _smooth_trajectory(pts_raw)

        fig.add_trace(go.Scatter3d(
            x=pts_smooth[:, 0],
            y=pts_smooth[:, 1],
            z=pts_smooth[:, 2],
            mode="lines",
            line=dict(color=color, width=2),
            opacity=0.55,
            name=f"Cluster {k}: {rep_truncated[k]}" if rep_truncated[k] else f"Cluster {k}",
            legendgroup=f"cluster_{k}",
            showlegend=show_legend,
            hoverinfo="skip",
        ))

    # Endpoint markers
    for c_idx in range(N_COMPLETIONS):
        k = int(labels[c_idx])
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ep = coords[endpoint_flat_idx[c_idx]]
        hover = completions[c_idx][:300]  # truncate for tooltip

        fig.add_trace(go.Scatter3d(
            x=[ep[0]], y=[ep[1]], z=[ep[2]],
            mode="markers",
            marker=dict(size=5, color=color, line=dict(color="white", width=0.5)),
            legendgroup=f"cluster_{k}",
            showlegend=False,
            hovertext=hover,
            hoverinfo="text",
            name=f"Cluster {k}",
        ))

    # Prompt origin marker
    fig.add_trace(go.Scatter3d(
        x=[origin[0]], y=[origin[1]], z=[origin[2]],
        mode="markers",
        marker=dict(
            size=12,
            symbol="diamond",
            color="black",
            line=dict(color="gold", width=3),
        ),
        name="Prompt origin",
        hovertext=PROMPT,
        hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(text=f'LLM Response Manifold — “{PROMPT}”', font=dict(size=15)),
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
        margin=dict(l=0, r=0, b=0, t=50),
    )
    return fig


# ===========================================================================
# 9. STATIC MATPLOTLIB 4-VIEW PNG
# ===========================================================================

def build_static_figure(
    coords: np.ndarray,
    comp_ids: list[int],
    snap_indices: list[int],
    completions: list[str],
    labels: np.ndarray,
    rep_truncated: list[str],
) -> None:
    print("[stage 7] Building static 4-view PNG …")

    # Rebuild per-completion arrays (same logic as Plotly section)
    traj_pts:  list[list[tuple[int, np.ndarray]]] = [[] for _ in range(N_COMPLETIONS)]
    endpoint_coords: list[np.ndarray] = [None] * N_COMPLETIONS
    origin = None

    for flat_i, (c_idx, s_idx) in enumerate(zip(comp_ids, snap_indices)):
        if c_idx == -1:
            origin = coords[flat_i]
        else:
            traj_pts[c_idx].append((s_idx, coords[flat_i]))
            if s_idx == N_SNAPSHOTS:
                endpoint_coords[c_idx] = coords[flat_i]

    for c_idx in range(N_COMPLETIONS):
        traj_pts[c_idx].sort(key=lambda x: x[0])

    view_angles = [
        (25,  45,  "Front-left"),
        (25, 135,  "Front-right"),
        (60,  45,  "Top-left"),
        (10,  10,  "Side"),
    ]

    fig = plt.figure(figsize=(14, 11))
    axes = []
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        axes.append(ax)

    legend_handles = []
    legend_labels  = []

    for ax_i, (ax, (elev, azim, view_name)) in enumerate(zip(axes, view_angles)):
        ax.set_title(view_name, fontsize=9, pad=4)
        ax.set_xlabel("MDS-1", fontsize=7, labelpad=2)
        ax.set_ylabel("MDS-2", fontsize=7, labelpad=2)
        ax.set_zlabel("MDS-3", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        legend_added = set()
        for c_idx in range(N_COMPLETIONS):
            k = int(labels[c_idx])
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            pts_raw = np.vstack([origin] + [p for _, p in traj_pts[c_idx]])
            pts_s   = _smooth_trajectory(pts_raw, n_out=80)

            line, = ax.plot(
                pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                color=color, alpha=0.45, linewidth=0.8,
            )
            if ax_i == 0 and k not in legend_added:
                legend_handles.append(line)
                legend_labels.append(f"C{k}: {rep_truncated[k][:45]}")
                legend_added.add(k)

            ep = endpoint_coords[c_idx]
            if ep is not None:
                ax.scatter(*ep, color=color, s=12, zorder=5, edgecolors="white", linewidths=0.3)

        # Prompt origin star
        star = ax.scatter(
            *origin, color="black", s=120, marker="*", zorder=10,
            edgecolors="gold", linewidths=0.8, label="Prompt origin",
        )
        if ax_i == 0:
            legend_handles.append(star)
            legend_labels.append("Prompt origin")

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.85,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f'LLM Response Manifold\n"{PROMPT}"',
        fontsize=11, y=0.99,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"          Saved {OUT_PNG}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print(" LLM RESPONSE-TRAJECTORY MANIFOLD VISUALISER")
    print("=" * 65)
    print(f" Prompt      : {PROMPT}")
    print(f" Model       : {GEN_MODEL}  |  Embed: {EMBED_MODEL}")
    print(f" Completions : {N_COMPLETIONS}  |  Snapshots/completion: {N_SNAPSHOTS}")
    print("=" * 65)

    # Stage 1 — completions
    completions = load_or_generate_completions()

    # Stage 2 — snapshots
    texts, comp_ids, snap_indices = build_snapshots(completions)

    # Stage 3 — embeddings
    embeddings = load_or_embed(texts)

    # Stage 4 — MDS
    coords = project_to_3d(embeddings, comp_ids, snap_indices)

    # Stage 5 — clustering
    labels, representatives, rep_truncated = cluster_endpoints(
        coords, comp_ids, snap_indices, completions
    )

    # Stage 6 — Plotly
    fig = build_plotly_figure(
        coords, comp_ids, snap_indices, completions, labels, rep_truncated
    )
    fig.write_html(OUT_HTML, include_plotlyjs=True)
    print(f"[stage 6] Saved interactive figure → {OUT_HTML}")

    # Stage 7 — Matplotlib static
    build_static_figure(
        coords, comp_ids, snap_indices, completions, labels, rep_truncated
    )
    print(f"[stage 7] Saved static figure       → {OUT_PNG}")

    print()
    print("Done.  Open", OUT_HTML, "in a browser to explore the manifold.")


if __name__ == "__main__":
    main()
