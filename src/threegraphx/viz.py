# src/threegraphx/viz.py
from __future__ import annotations

from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt


# ------------------------------ helpers ------------------------------ #

_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}
_DEFAULT_SYMBOL = "X"


def _to_numpy(x) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _project_pos(pos_np: np.ndarray) -> np.ndarray:
    """Project N×D (D>=2) to N×2. If D==2, return as-is. If D>=3, do PCA(2).
    If anything goes wrong, fall back to first two cols."""
    if pos_np is None:
        return None
    if pos_np.ndim != 2 or pos_np.shape[1] < 2:
        raise ValueError("pos must be [N,2+] for plotting")
    if pos_np.shape[1] == 2:
        return pos_np.astype(float)
    # PCA to 2D without sklearn
    X = pos_np.astype(float)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = np.dot(Xc.T, Xc) / max(1, Xc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:2]
    W = vecs[:, idx]
    return Xc @ W


def _edge_index_to_list(edge_index) -> List[Tuple[int, int]]:
    """Accept [2,E] tensor/array or list of tuples; return unique undirected list."""
    if edge_index is None:
        return []
    if isinstance(edge_index, (torch.Tensor, np.ndarray)):
        ei = _to_numpy(edge_index)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must be shape [2, E]")
        edges = []
        for u, v in zip(ei[0], ei[1]):
            a, b = int(u), int(v)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.append((a, b))
        return sorted(list(set(edges)))
    # assume list/iter of pairs
    edges = []
    for u, v in edge_index:
        a, b = int(u), int(v)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edges.append((a, b))
    return sorted(list(set(edges)))


def _labels_from_z(z_np: Optional[np.ndarray], N: int) -> List[str]:
    """Return ['idx:symbol'] labels. If z missing/unknown, use X."""
    labels = []
    for i in range(N):
        if z_np is None:
            sym = _DEFAULT_SYMBOL
        else:
            zval = int(z_np[i])
            sym = _SYMBOLS.get(zval, _DEFAULT_SYMBOL)
        labels.append(f"{i}:{sym}")
    return labels


def _build_graph(N: int, edge_index) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for u, v in _edge_index_to_list(edge_index):
        G.add_edge(u, v)
    return G


def _safe_pos(
    G: nx.Graph, pos_2d: Optional[np.ndarray]
) -> Dict[int, Tuple[float, float]]:
    if pos_2d is None or pos_2d.shape[0] != G.number_of_nodes():
        return nx.spring_layout(G, seed=0)
    return {i: (float(pos_2d[i, 0]), float(pos_2d[i, 1])) for i in G.nodes()}


# ------------------------------ public API ------------------------------ #


def visualize_graph(
    *,
    z=None,
    pos=None,
    edge_index=None,
    ax: Optional[plt.Axes] = None,
    title: str = "Graph",
    node_size: int = 300,
    save_path: Optional[str] = None,
):
    """
    Whole-graph view with node index and element type.

    Args:
        z: [N] atomic numbers (Tensor/ndarray/list) or None -> 'X'
        pos: [N, 2 or 3] positions or None -> spring layout
        edge_index: [2,E] or list of (u,v)
        ax: matplotlib Axes, or None to create a new figure
        title, node_size, save_path: styling
    """
    z_np = _to_numpy(z) if z is not None else None
    N = (
        len(z_np)
        if z_np is not None
        else (
            max(max(e) for e in _edge_index_to_list(edge_index)) + 1
            if edge_index is not None
            else 0
        )
    )
    if N == 0:
        raise ValueError("Cannot infer number of nodes; provide z or edge_index.")

    G = _build_graph(N, edge_index)
    pos_np = _to_numpy(pos)
    pos_2d = _project_pos(pos_np) if pos_np is not None else None
    pos_map = _safe_pos(G, pos_2d)

    labels = _labels_from_z(z_np, N)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    nx.draw_networkx_edges(G, pos_map, ax=ax, edge_color="#777777", width=1.5)
    nx.draw_networkx_nodes(
        G,
        pos_map,
        ax=ax,
        node_color="#cfd8dc",
        node_size=node_size,
        linewidths=0.8,
        edgecolors="#455a64",
    )
    nx.draw_networkx_labels(
        G, pos_map, ax=ax, labels={i: labels[i] for i in G.nodes()}, font_size=9
    )

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return ax


def visualize_mask(
    *,
    z=None,
    pos=None,
    edge_index=None,
    mask=None,  # [N] or [N,1] torch/np
    threshold: float = 0.5,  # binary explanatory cutoff
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    title: str = "Explanation (node mask)",
    node_size: int = 320,
    show_colorbar: bool = True,
    save_path: Optional[str] = None,
):
    """
    Heatmap of node importance with binary explanatory overlay.

    - Node facecolor = continuous mask value via cmap.
    - Node border color: highlighted if mask>=threshold.
    - Edge width: thicker if both endpoints explanatory.
    """
    mask_np = _to_numpy(mask)
    if mask_np is None:
        raise ValueError("mask is required")
    if mask_np.ndim == 2 and mask_np.shape[1] == 1:
        mask_np = mask_np[:, 0]
    if mask_np.ndim != 1:
        raise ValueError("mask must be [N] or [N,1]")

    N = mask_np.shape[0]
    z_np = _to_numpy(z) if z is not None else None

    G = _build_graph(N, edge_index)
    pos_np = _to_numpy(pos)
    pos_2d = _project_pos(pos_np) if pos_np is not None else None
    pos_map = _safe_pos(G, pos_2d)

    labels = _labels_from_z(z_np, N)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))

    # Normalize mask to [0,1] safely
    m = mask_np.astype(float)
    m = np.clip(m, 0.0, 1.0)
    explanatory = m >= float(threshold)

    # Edges: width based on both endpoints explanatory
    widths = []
    for u, v in G.edges():
        w = 2.5 if explanatory[u] and explanatory[v] else 1.0
        widths.append(w)
    nx.draw_networkx_edges(G, pos_map, ax=ax, edge_color="#8a8a8a", width=widths)

    # Nodes: facecolor from cmap, outline if explanatory
    sc = nx.draw_networkx_nodes(
        G,
        pos_map,
        ax=ax,
        node_color=m,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        node_size=node_size,
        linewidths=np.where(explanatory, 2.3, 0.8),
        edgecolors=np.where(explanatory, "#222222", "#6d6d6d"),
    )
    nx.draw_networkx_labels(
        G, pos_map, ax=ax, labels={i: labels[i] for i in G.nodes()}, font_size=8
    )

    if show_colorbar:
        cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
        cb.set_label("Node importance (mask)")

    ax.set_title(title + f"  (threshold={threshold:.2f})")
    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    return ax


def visualize_clusters(
    *,
    z=None,
    pos=None,
    edge_index=None,
    clusters: List[List[int]],
    ax: Optional[plt.Axes] = None,
    title: str = "Clusters",
    node_size: int = 320,
    save_path: Optional[str] = None,
):
    """
    Color each cluster; overlapping nodes get a thick black outline.
    """
    N = (
        len(z)
        if isinstance(z, (list, np.ndarray, torch.Tensor))
        else max(max(e) for e in _edge_index_to_list(edge_index)) + 1
    )
    z_np = _to_numpy(z) if z is not None else None

    G = _build_graph(N, edge_index)
    pos_np = _to_numpy(pos)
    pos_2d = _project_pos(pos_np) if pos_np is not None else None
    pos_map = _safe_pos(G, pos_2d)

    labels = _labels_from_z(z_np, N)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.6))

    # Assign colors per cluster
    cmap = plt.get_cmap("tab20")
    node_colors = np.full(N, -1, dtype=int)
    node_counts = np.zeros(N, dtype=int)
    for cid, cl in enumerate(clusters):
        for a in cl:
            node_colors[a] = cid % 20
            node_counts[a] += 1

    facecolors = []
    for i in range(N):
        if node_colors[i] >= 0:
            facecolors.append(cmap(node_colors[i]))
        else:
            facecolors.append((0.82, 0.85, 0.88, 1.0))  # light gray for non-cluster

    # Overlaps = nodes in multiple clusters
    outline = np.where(node_counts > 1, "#000000", "#455a64")
    lw = np.where(node_counts > 1, 2.6, 0.9)

    nx.draw_networkx_edges(G, pos_map, ax=ax, edge_color="#8a8a8a", width=1.4)
    nx.draw_networkx_nodes(
        G,
        pos_map,
        ax=ax,
        node_color=facecolors,
        node_size=node_size,
        linewidths=lw,
        edgecolors=outline,
    )
    nx.draw_networkx_labels(
        G, pos_map, ax=ax, labels={i: labels[i] for i in G.nodes()}, font_size=8
    )

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    return ax


def visualize_explanatory_subgraph(
    *,
    z=None,
    pos=None,
    edge_index=None,
    mask=None,
    threshold: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: str = "Explanatory Subgraph",
    node_size: int = 340,
    save_path: Optional[str] = None,
):
    """
    Show only nodes with mask>=threshold and the induced edges.
    """
    mask_np = _to_numpy(mask)
    if mask_np is None:
        raise ValueError("mask is required")
    if mask_np.ndim == 2 and mask_np.shape[1] == 1:
        mask_np = mask_np[:, 0]
    if mask_np.ndim != 1:
        raise ValueError("mask must be [N] or [N,1]")
    N = mask_np.shape[0]

    keep = mask_np >= float(threshold)
    if keep.sum() == 0:
        raise ValueError("No nodes pass the threshold; try lowering it.")

    idx_map = {i: k for k, i in enumerate(np.where(keep)[0])}
    # Build subgraph edges:
    E = _edge_index_to_list(edge_index)
    sub_edges = [(idx_map[u], idx_map[v]) for (u, v) in E if keep[u] and keep[v]]

    z_np = _to_numpy(z) if z is not None else None
    z_sub = z_np[keep] if z_np is not None else None
    pos_np = _to_numpy(pos)
    pos_sub = pos_np[keep] if pos_np is not None else None

    G = _build_graph(int(keep.sum()), sub_edges)
    pos_2d = _project_pos(pos_sub) if pos_sub is not None else None
    pos_map = _safe_pos(G, pos_2d)

    labels = _labels_from_z(z_sub, G.number_of_nodes())
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 4.8))

    nx.draw_networkx_edges(G, pos_map, ax=ax, edge_color="#666666", width=2.2)
    nx.draw_networkx_nodes(
        G,
        pos_map,
        ax=ax,
        node_color="#ffcc80",
        node_size=node_size,
        linewidths=1.5,
        edgecolors="#e65100",
    )
    nx.draw_networkx_labels(
        G, pos_map, ax=ax, labels={i: labels[i] for i in G.nodes()}, font_size=9
    )

    ax.set_title(title + f"  (threshold={threshold:.2f})")
    ax.set_axis_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    return ax
