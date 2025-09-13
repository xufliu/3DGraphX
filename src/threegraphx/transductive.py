from __future__ import annotations
from typing import Optional, Union, List
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.explain import ExplainerAlgorithm, Explanation
from torch_geometric.explain.config import MaskType, ModelMode
from torch_geometric.nn import knn_graph

from .hooks.base import BackboneHooks, PreparedBatch, MaskPoint
from .utils import tree_decompose, set_masks, clear_masks


class GraphXTransductive(ExplainerAlgorithm):
    """
    Transductive explainer:
      - Learns a cluster-level mask inside `forward` on the *current* graph.
      - Applies the (soft) atom-level mask at a chosen MaskPoint via BackboneHooks.
    """

    coeffs = {
        "edge_size": 0.005,
        "edge_reduction": "sum",
        "node_feat_size": 1.0,
        "node_feat_reduction": "mean",
        "edge_ent": 1.0,
        "node_feat_ent": 0.1,
        "EPS": 1e-15,
    }

    def __init__(
        self,
        hooks: BackboneHooks,
        epochs: int = 100,
        lr: float = 1e-2,
        mask_point: MaskPoint = MaskPoint.EMBED,
        **kwargs,
    ):
        super().__init__()
        self.hooks = hooks
        self.epochs = epochs
        self.lr = lr
        self.mask_point = mask_point
        self.coeffs.update(kwargs)

        # trainable masks
        self.node_mask: Optional[Parameter] = (
            None  # cluster-level parameter [C,1] (logits)
        )
        self.edge_mask: Optional[Parameter] = (
            None  # optional edge mask (if enabled via config)
        )

        # hard (support) masks derived from first backward pass
        self.hard_node_mask = None
        self.hard_edge_mask = None

        # bookkeeping for outputs
        self._clusters = None  # list[list[int]]
        self._atom_mask = None  # [N,1] atom-level mask used during training
        self._cluster_to_atom_T = None  # [C,N] transform matrix
        self._captures = None  # optional dict with "embed" / "pre_agg"

    def supports(self) -> bool:
        return True

    @torch.no_grad()
    def _build_clusters(self, z: Tensor, pos: Tensor, edges_np=None):
        """
        Build tree-based clusters used to parameterize the mask.
        If `edges_np` is None, we form a 2-NN graph on coordinates.
        """
        if edges_np is None:
            e = knn_graph(pos, k=2).cpu().numpy()
        else:
            e = edges_np

        edges = set()
        for s, t in zip(e[0], e[1]):
            a, b = int(s), int(t)
            if a > b:
                a, b = b, a
            if a != b:
                edges.add((a, b))
        edges = sorted(edges)
        clusters, _rot = tree_decompose(z, edges)
        return clusters

    def _cluster_transform(self, clusters: List[List[int]], num_nodes: int) -> Tensor:
        """
        Build a [C, N] transform that evenly distributes a cluster's mass
        over its atoms. Multiple-cluster membership is handled by 1/deg.
        """
        T = torch.zeros(
            len(clusters),
            num_nodes,
            dtype=torch.float32,
            device="cuda" if self._is_cuda else "cpu",
        )
        from collections import defaultdict

        atom_to_cs = defaultdict(list)
        for ci, cl in enumerate(clusters):
            for a in cl:
                atom_to_cs[a].append(ci)
        for a, cs in atom_to_cs.items():
            w = 1.0 / max(1, len(cs))
            for ci in cs:
                T[ci, a] = w
        return T

    @property
    def _is_cuda(self) -> bool:
        return torch.cuda.is_available()

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        """
        x: usually atomic numbers z (shape [N])
        edge_index: usually positions pos (shape [N,3]) in this 3D-chemistry setting
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs not supported in '{self.__class__.__name__}'"
            )

        # one-shot training on this graph
        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        # Expose atom-level mask for visualization; cluster-level mask for debug
        atom_mask = self._atom_mask  # [N,1]
        cluster_mask = (
            torch.sigmoid(self.node_mask) if self.node_mask is not None else None
        )

        # Optional post-processing for edge_mask (if used)
        _ = self._post_process_mask(
            self.edge_mask, self.hard_edge_mask, apply_sigmoid=True
        )

        clear_masks(model)  # remove any hooks on the backbone

        # The PyG Explanation container is flexible; we attach what users need.
        return Explanation(
            node_mask=atom_mask,  # [N,1] primary mask
            cluster_mask=cluster_mask,  # [C,1] cluster logits -> prob
            clusters=self._clusters,  # list of clusters
            captures=self._captures or {},  # optional: {"embed":..., "pre_agg":...}
        )

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> None:
        model.eval()  # we only optimize mask params; backbone remains fixed

        # Prepare inputs via hooks (x=z, edge_index=pos convention):
        prep: PreparedBatch = self.hooks.prepare_inputs(x, edge_index, **kwargs)

        # Build clusters (from provided edges if any, else 2-NN on pos):
        edges_np = kwargs.pop("edges", None)
        clusters = self._build_clusters(prep.z, prep.pos, edges_np)
        self._clusters = clusters

        # Build backbone message-passing edges:
        model_edge_index = self.hooks.make_edge_index(model, prep)

        # Initialize learnable masks (cluster-level node mask; edge mask optional):
        self._initialize_masks(
            x=prep.z, mp_edge_index=model_edge_index, num_clusters=len(clusters)
        )

        # Register edge mask (if any) on the model parameters:
        parameters: List[Parameter] = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(
                model, self.edge_mask, edge_index=model_edge_index, apply_sigmoid=True
            )
            parameters.append(self.edge_mask)

        # Cluster -> atom transform [C, N]:
        T = self._cluster_transform(clusters, num_nodes=prep.z.size(0))
        T = T.to(prep.z.device)
        self._cluster_to_atom_T = T

        # Atom-level mask used for training (soft probabilities in [0,1]):
        node_mask_cluster = (
            torch.sigmoid(self.node_mask) if self.node_mask is not None else None
        )
        if node_mask_cluster is None:
            raise RuntimeError("node_mask is not initialized")
        node_mask_atom = (node_mask_cluster.T @ T).T  # [N,1]

        # Cache optimizer:
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        # Build extras (edge features, etc.) via hooks:
        prep = prep.with_(
            edge_index=model_edge_index,
            extras=self.hooks.edge_attr_and_extras(model, prep),
        )

        # Train loop (on this graph):
        last_captures = None
        for it in range(self.epochs):
            optimizer.zero_grad()

            raw_out, captures = self.hooks.backbone_step(
                model, prep, node_mask_atom, where=self.mask_point
            )
            out = self.hooks.readout(model, raw_out, prep).view(-1)

            y_hat, y = out, target
            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)
            loss.backward(retain_graph=True)
            optimizer.step()

            if it == 0:
                # Collect support (hard) masks for regularization terms
                if self.node_mask.grad is None:
                    raise ValueError(
                        "No gradients for node mask; ensure the mask influences the model."
                    )
                self.hard_node_mask = self.node_mask.grad != 0.0

                if self.edge_mask is not None:
                    if self.edge_mask.grad is None:
                        raise ValueError(
                            "No gradients for edge mask; ensure edges influence the model."
                        )
                    self.hard_edge_mask = self.edge_mask.grad != 0.0

            last_captures = captures

        # Store final artifacts:
        self._captures = last_captures
        self._atom_mask = node_mask_atom  # [N,1]

    def _initialize_masks(self, x: Tensor, mp_edge_index: Tensor, num_clusters: int):
        """
        Initialize learnable masks based on the explainer_config.
        Node mask is parameterized over clusters (C).
        Edge mask is optional (E).
        """
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        E = mp_edge_index.size(1)
        C = max(1, num_clusters)
        F = 1
        std = 0.1

        # Node (cluster) mask:
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(C, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(C, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            raise AssertionError(f"Unsupported node_mask_type: {node_mask_type}")

        # Edge mask (optional):
        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            gain = torch.nn.init.calculate_gain("relu")
            std_e = gain * sqrt(2.0 / (2 * C))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std_e)
        else:
            raise AssertionError(f"Unsupported edge_mask_type: {edge_mask_type}")

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        # Data-fit loss:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = F.l1_loss(y_hat, y)
        else:
            raise AssertionError(f"Unsupported model mode: {self.model_config.mode}")

        # Regularize (optional) edge mask:
        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs["edge_reduction"])
            loss = loss + self.coeffs["edge_size"] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs["EPS"]) - (1 - m) * torch.log(
                1 - m + self.coeffs["EPS"]
            )
            loss = loss + self.coeffs["edge_ent"] * ent.mean()

        # Regularize cluster node mask:
        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs["node_feat_reduction"])
            loss = loss + self.coeffs["node_feat_size"] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs["EPS"]) - (1 - m) * torch.log(
                1 - m + self.coeffs["EPS"]
            )
            loss = loss + self.coeffs["node_feat_ent"] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
