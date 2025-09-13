# src/threegraphx/utils.py
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any, Sequence

import numpy as np
import networkx as nx
import torch

# Re-export thin wrappers so callers use a single utils import:
try:
    from torch_geometric.explain.algorithm.utils import (
        set_masks as _pyg_set_masks,
        clear_masks as _pyg_clear_masks,
    )
except Exception as e:  # pragma: no cover
    _pyg_set_masks = None
    _pyg_clear_masks = None


def set_masks(model, mask, **kwargs):
    """
    Thin wrapper over PyG's set_masks; kept here to avoid deep imports
    throughout the project.
    """
    if _pyg_set_masks is None:
        raise RuntimeError("PyG set_masks not available; check your installation.")
    return _pyg_set_masks(model, mask, **kwargs)


def clear_masks(model):
    """
    Thin wrapper over PyG's clear_masks; kept here to avoid deep imports
    throughout the project.
    """
    if _pyg_clear_masks is None:
        raise RuntimeError("PyG clear_masks not available; check your installation.")
    return _pyg_clear_masks(model)


# --------------------------------------------------------------------------- #
# Graph motif clustering (tree_decompose)
# --------------------------------------------------------------------------- #


def _infer_num_nodes(atom_names: Any, all_edges: Sequence[Tuple[int, int]]) -> int:
    """
    Try to infer number of nodes robustly.

    Accepts:
      - torch.Tensor (e.g., z)  -> len(z)
      - np.ndarray              -> len(arr)
      - int N                   -> N
      - list/tuple of indices   -> max index + 1 (fallback)
      - otherwise               -> from edges (max index + 1)

    Returns:
      N (int)
    """
    if isinstance(atom_names, torch.Tensor):
        return int(atom_names.size(0))
    if isinstance(atom_names, np.ndarray):
        return int(atom_names.shape[0])
    if isinstance(atom_names, int):
        return atom_names
    if (
        isinstance(atom_names, (list, tuple))
        and len(atom_names) > 0
        and isinstance(atom_names[0], (int, np.integer))
    ):
        return int(max(atom_names) + 1)
    if all_edges:
        m = max(int(u) for e in all_edges for u in e)
        return m + 1
    raise ValueError(
        "Cannot infer number of nodes; provide a tensor/array/int or non-empty edges."
    )


def tree_decompose(
    atom_names: Any, all_edges: Sequence[Tuple[int, int]]
) -> Tuple[List[List[int]], Dict[int, bool]]:
    """
    Decompose a molecular graph into simple clusters (acyclic edges + rings).

    Differences vs the original snippet:
      * Always interprets nodes as indices 0..N-1 (not atomic numbers).
      * Safer merging of adjacent leaf-edge clusters.
      * Returns clusters as List[List[int]] and a dict `rotable` marking
        2-atom clusters whose endpoints both have degree >= 2.

    Args:
      atom_names: Usually z (Tensor[N]) or N, used only to infer N.
      all_edges: Iterable of (u, v) with 0-based integer indices.

    Returns:
      clusters: list of clusters, each a list[int] of atom indices.
      rotable:  dict mapping cluster_id -> bool (True if rotatable bond).
    """
    # Normalize edges to 0-based ints and deduplicate:
    edges_set = set()
    for u, v in all_edges:
        a, b = int(u), int(v)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        edges_set.add((a, b))
    edges = sorted(edges_set)

    # Build graph with indices 0..N-1
    N = _infer_num_nodes(atom_names, edges)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    # Rings via cycle basis:
    ring_list = nx.cycle_basis(G)
    ring_list = sorted(ring_list, key=len)  # small rings first; not essential

    # Start with all non-ring edges as seed clusters of size 2:
    clusters: List[set[int]] = []
    ring_edges = set()
    for cyc in ring_list:
        cyc_seq = list(cyc) + [cyc[0]]
        for i in range(len(cyc)):
            a, b = sorted((cyc_seq[i], cyc_seq[i + 1]))
            ring_edges.add((a, b))

    for u, v in G.edges:
        if (u, v) not in ring_edges and (v, u) not in ring_edges:
            clusters.append({u, v})

    # Merge adjacent leaf-leaf edges that share an atom (extend linear chains):
    # If two edge-clusters share a single atom, and both "outer" atoms
    # have degree 1, merge them into one cluster.
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if not clusters[i]:
                continue
            for j in range(i + 1, len(clusters)):
                if not clusters[j]:
                    continue
                inter = clusters[i] & clusters[j]
                if len(inter) != 1:
                    continue
                # endpoints (atoms not in the intersection) of the two edges:
                ai = next(iter(clusters[i] - inter))
                aj = next(iter(clusters[j] - inter))
                # Merge only if both are leaves in G:
                if G.degree(ai) == 1 and G.degree(aj) == 1:
                    clusters[i] |= clusters[j]
                    clusters[j] = set()
                    changed = True

    # Drop empties and convert to lists:
    edge_clusters: List[List[int]] = [sorted(list(c)) for c in clusters if c]

    # Append rings as their own clusters:
    for cyc in ring_list:
        edge_clusters.append(sorted(list(cyc)))

    # Rotatable flag: only for 2-atom clusters whose endpoints both have degree >= 2
    rotable: Dict[int, bool] = {}
    for idx, cl in enumerate(edge_clusters):
        rot = False
        if len(cl) == 2:
            u, v = cl
            if G.degree(u) >= 2 and G.degree(v) >= 2:
                rot = True
        rotable[idx] = rot

    return edge_clusters, rotable
