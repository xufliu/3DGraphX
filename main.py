# main.py
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric.datasets import QM9
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.nn import knn_graph
from torch_geometric.nn.models.schnet import SchNet
from torch_geometric.nn.models.dimenet import DimeNet, DimeNetPlusPlus

HERE = osp.dirname(osp.realpath(__file__))
SRC = osp.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from threegraphx.hooks.base import MaskPoint
from threegraphx.hooks.schnet import SchNetHooks
from threegraphx.hooks.dimenet import DimeNetHooks
from threegraphx.transductive import GraphXTransductive
from threegraphx.inductive import GraphXInductive


# --------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="ThreeGraphX on QM9 with SchNet or DimeNet(++)"
    )

    # Backbone
    p.add_argument(
        "--backbone",
        choices=["schnet", "dimenet"],
        default="schnet",
        help="Choose the underlying GNN.",
    )
    # DimeNet vs DimeNet++
    p.add_argument("--use-dimenet-plus-plus", dest="dpp", action="store_true")
    p.add_argument("--no-dimenet-plus-plus", dest="dpp", action="store_false")
    p.set_defaults(dpp=True)  # default: DimeNet++

    # Explainer
    p.add_argument(
        "--explainer", choices=["transductive", "inductive"], default="inductive"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument(
        "--mask-point",
        choices=["embed", "pre_agg"],
        default="embed",
        help="Where to apply the node mask inside backbone.",
    )
    p.add_argument(
        "--feature-point",
        choices=["embed", "pre_agg"],
        default="embed",
        help="Where to read node features for cluster pooling (inductive only).",
    )

    # Dataset/target
    p.add_argument(
        "--target-attr", type=int, default=0, help="QM9 property index (0..11)."
    )
    p.add_argument("--data-root", default=osp.join(HERE, "data", "QM9"))
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--test-size", type=int, default=1024)

    # Training/eval
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint",
        default="checkpoints/3dgraphx.pth",
        help="Explainer weights (inductive only).",
    )
    p.add_argument(
        "--topk",
        default="2,3,4,5,6,7,8,9",
        help="Comma-separated list of k for evaluation.",
    )

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def maskpoint_from_str(s: str) -> MaskPoint:
    return MaskPoint.EMBED if s.lower() == "embed" else MaskPoint.PRE_AGG


# --------------------------------------------------------------------- #
# Backbone + data builders
# --------------------------------------------------------------------- #
def build_schnet_and_data(device: str, data_root: str, target_attr: int):
    ds = QM9(data_root)
    model, datasets = SchNet.from_qm9_pretrained(data_root, ds, target_attr)
    model = model.to(device)
    return model, datasets  # (train, val, test)


def _split_dataset(ds: QM9, train_size: int, val_size: int, test_size: int, seed: int):
    N = len(ds)
    total = train_size + val_size + test_size
    if total > N:
        raise ValueError(f"Requested sizes ({total}) exceed dataset size ({N}).")
    # Deterministic split:
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, lengths=[train_size, val_size, test_size], generator=g)


def build_dimenet_and_data(
    device: str,
    data_root: str,
    target_attr: int,
    use_pp: bool,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
):
    """
    Follow PyG DimeNet example style:
      - Subselect QM9 targets to 12 entries via 'idx'
      - Construct DimeNet or DimeNet++
    """
    dataset = QM9(data_root)

    # Subselect labels as in common DimeNet examples:
    # (0,1,2,3,4,5,6,12,13,14,15,11) => 12 targets
    idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
    dataset.data.y = dataset.data.y[:, idx]

    # Build backbone (untrained by default; you may load a separate checkpoint)
    if use_pp:
        model = DimeNetPlusPlus(out_channels=1).to(device)
    else:
        model = DimeNet(out_channels=1).to(device)

    # Deterministic split to the requested sizes:
    train_ds, val_ds, test_ds = _split_dataset(
        dataset, train_size, val_size, test_size, seed
    )
    return model, (train_ds, val_ds, test_ds)


# --------------------------------------------------------------------- #
# Explainer factory
# --------------------------------------------------------------------- #
def build_explainer(args, model, hooks) -> Explainer:
    if args.explainer == "transductive":
        algo = GraphXTransductive(
            hooks=hooks,
            epochs=args.epochs,
            lr=args.lr,
            mask_point=maskpoint_from_str(args.mask_point),
        )
        node_mask_type = "object"  # cluster-parameterized
        edge_mask_type = None
    else:
        algo = GraphXInductive(
            hooks=hooks,
            epochs=args.epochs,
            lr=args.lr,
            mask_point=maskpoint_from_str(args.mask_point),
            feature_point=maskpoint_from_str(args.feature_point),
        )
        node_mask_type = None
        edge_mask_type = None

    return Explainer(
        model=model,
        algorithm=algo,
        explanation_type="phenomenon",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=ModelConfig(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )


# --------------------------------------------------------------------- #
# Checkpoint helpers (inductive only)
# --------------------------------------------------------------------- #
def maybe_load_checkpoint(explainer: Explainer, ckpt_path: str) -> int:
    if not osp.exists(ckpt_path):
        print("No checkpoint; starting from scratch.")
        return 0
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        explainer.algorithm.load_state_dict(ckpt["model_state_dict"])
        if (
            hasattr(explainer.algorithm, "optimizer")
            and explainer.algorithm.optimizer is not None
        ):
            explainer.algorithm.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 0


def maybe_save_checkpoint(explainer: Explainer, epoch: int, ckpt_path: str):
    if not hasattr(explainer.algorithm, "optimizer"):
        return
    os.makedirs(osp.dirname(ckpt_path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": explainer.algorithm.state_dict(),
            "optimizer_state_dict": (
                explainer.algorithm.optimizer.state_dict()
                if explainer.algorithm.optimizer is not None
                else {}
            ),
        },
        ckpt_path,
    )


# --------------------------------------------------------------------- #
# Eval helpers
# --------------------------------------------------------------------- #
def edge_index_numpy_from_knn(pos: torch.Tensor, k: int = 2) -> np.ndarray:
    ei = knn_graph(pos, k=k)
    return ei.detach().cpu().numpy()


def topk_atoms_from_cluster_mask(
    cluster_scores: torch.Tensor, clusters: List[List[int]], k: int
) -> List[int]:
    scores = cluster_scores.detach().view(-1)
    order = torch.argsort(scores, descending=True).tolist()
    chosen: List[int] = []
    for cid in order:
        for a in clusters[cid]:
            if a not in chosen:
                chosen.append(a)
            if len(chosen) >= k:
                return chosen
    return chosen[:k]


def eval_on_topk(
    model,
    z: torch.Tensor,
    pos: torch.Tensor,
    node_mask: torch.Tensor,
    clusters: Optional[List[List[int]]] = None,
    cluster_scores: Optional[torch.Tensor] = None,
    ks: List[int] = [2, 3, 4, 5],
) -> dict:
    with torch.no_grad():
        full = model(z, pos)[0]
    m = node_mask.detach().view(-1)
    order_nodes = torch.argsort(m, descending=True).tolist()

    out = {}
    for k in ks:
        idx_nodes = sorted(order_nodes[:k])
        pred_n = model(z[idx_nodes], pos[idx_nodes])[0]
        loss_n = F.l1_loss(pred_n, full).item()

        loss_c = None
        if clusters is not None and cluster_scores is not None:
            chosen = topk_atoms_from_cluster_mask(cluster_scores, clusters, k)
            idx_c = sorted(chosen[:k])
            pred_c = model(z[idx_c], pos[idx_c])[0]
            loss_c = F.l1_loss(pred_c, full).item()

        out[k] = (loss_n, loss_c)
    return out


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    # Build backbone + datasets
    if args.backbone == "schnet":
        hooks = SchNetHooks()
        model, datasets = build_schnet_and_data(
            args.device, args.data_root, args.target_attr
        )
        train_dataset, val_dataset, test_dataset = datasets
    else:
        hooks = DimeNetHooks()
        model, (train_dataset, val_dataset, test_dataset) = build_dimenet_and_data(
            device=args.device,
            data_root=args.data_root,
            target_attr=args.target_attr,
            use_pp=args.dpp,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )

    # Optionally clip sizes again (harmless if already exact from DimeNet path)
    train_dataset = train_dataset[: args.train_size]
    val_dataset = val_dataset[: args.val_size]
    test_dataset = test_dataset[: args.test_size]

    # Build explainer
    explainer = build_explainer(args, model, hooks)

    # Train (only for inductive)
    start_epoch = 0
    if args.explainer == "inductive":
        start_epoch = maybe_load_checkpoint(explainer, args.checkpoint)
        for epoch in range(start_epoch, args.epochs):
            running = []
            for data in train_dataset:
                data = data.to(args.device)
                edges_np = edge_index_numpy_from_knn(data.pos, k=2)
                loss = explainer.algorithm.fit_epoch(
                    epoch,
                    model,
                    x=data.z,
                    edge_index=data.pos,  # our hooks treat edge_index as pos
                    target=data.y[:, args.target_attr],
                    edges=edges_np,
                )
                running.append(loss)
            print(f"[epoch {epoch:03d}] loss={np.mean(running):.4f}")
            if (epoch + 1) % 5 == 0:
                maybe_save_checkpoint(explainer, epoch, args.checkpoint)

    # Evaluate on test by top-k
    topk_list = [int(s) for s in args.topk.split(",") if s.strip()]
    ass_nodes = {k: [] for k in topk_list}
    ass_clust = {k: [] for k in topk_list}

    for i, data in enumerate(test_dataset, start=0):
        data = data.to(args.device)
        edges_np = edge_index_numpy_from_knn(data.pos, k=2)

        explanation = explainer(
            x=data.z,
            edge_index=data.pos,  # pos
            target=data.y[:, args.target_attr].double(),
            edges=edges_np,
        )

        node_mask = explanation.node_mask
        if node_mask.dim() == 1:
            node_mask = node_mask.view(-1, 1)

        cluster_scores = None
        clusters = getattr(explanation, "clusters", None)
        # Name harmonization across explainers:
        if hasattr(explanation, "motif_mask"):  # inductive
            cluster_scores = explanation.motif_mask
        elif hasattr(explanation, "cluster_mask"):  # if you add it in transductive
            cluster_scores = explanation.cluster_mask

        res = eval_on_topk(
            model,
            data.z,
            data.pos,
            node_mask=node_mask,
            clusters=clusters,
            cluster_scores=cluster_scores,
            ks=topk_list,
        )
        for k, (ln, lc) in res.items():
            ass_nodes[k].append(ln)
            if lc is not None:
                ass_clust[k].append(lc)

        if i < 5:
            print(f"[example {i}] { {k: res[k] for k in topk_list} }")

    print("\n=== Evaluation (mean ± std L1 to full prediction) ===")
    for k in topk_list:
        mn, sd = np.mean(ass_nodes[k]), np.std(ass_nodes[k])
        if len(ass_clust[k]) > 0:
            mnc, sdc = np.mean(ass_clust[k]), np.std(ass_clust[k])
            print(
                f"k={k:2d}  nodes: {mn:.4f} ± {sd:.4f}   clusters: {mnc:.4f} ± {sdc:.4f}"
            )
        else:
            print(f"k={k:2d}  nodes: {mn:.4f} ± {sd:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
