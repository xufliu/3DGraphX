# src/threegraphx/inductive.py
from __future__ import annotations

from typing import Optional, Union, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.explain import ExplainerAlgorithm, Explanation
from torch_geometric.explain.config import ModelMode
from torch_geometric.nn import knn_graph

from .hooks.base import BackboneHooks, PreparedBatch, MaskPoint
from .utils import tree_decompose, clear_masks  # expected in your utils


class GraphXInductive(ExplainerAlgorithm):
    """
    Inductive (trainable) explainer.

    - Trains an MLP over cluster embeddings to predict cluster logits.
    - Samples (Concrete) soft cluster masks during training; converts to
      atom-level masks and applies them in the backbone via hooks.
    - At inference, produces sigmoid(cluster logits) without sampling.

    Key knobs:
      * mask_point: where the mask is applied inside the backbone (EMBED or PRE_AGG)
      * feature_point: which node representation to pool for cluster embeddings
                       ("embed" earliest or "pre_agg" right before readout)
    """

    coeffs = {
        "edge_size": 0.05,  # L1 on mask (size)
        "edge_ent": 1.0,  # entropy reg
        "temp": [5.0, 2.0],  # Gumbel-Softmax temperature schedule [t0, t1]
        "bias": 0.0,  # sampling bias in Concrete
    }

    def __init__(
        self,
        hooks: BackboneHooks,
        epochs: int = 100,
        lr: float = 3e-3,
        mask_point: MaskPoint = MaskPoint.EMBED,
        feature_point: MaskPoint = MaskPoint.EMBED,
        hidden: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.hooks = hooks
        self.epochs = epochs
        self.lr = lr
        self.mask_point = mask_point
        self.feature_point = feature_point
        self.hidden = hidden
        self.coeffs.update(kwargs)

        # MLP for cluster -> logit (lazy init once input dim is known)
        self.mlp: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # training bookkeeping
        self._curr_epoch = -1
        self._cached_clusters: Optional[List[List[int]]] = None

    # --------------------------------------------------------------------- #
    # Public training API
    # --------------------------------------------------------------------- #
    def fit_epoch(
        self,
        epoch: int,
        model: nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> float:
        """
        Train the explainer's MLP for a single epoch on a single graph batch.

        Args:
            epoch: current epoch id (0-based)
            model: frozen backbone (SchNet/DimeNet/...)
            x: typically atomic numbers z [N]
            edge_index: typically positions pos [N,3]
            target: scalar or per-graph target
            index: optional index for node/graph selection
            kwargs: may contain 'edges' (numpy KNN edges) and 'batch'
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs not supported in '{self.__class__.__name__}'"
            )

        model.eval()

        # Prepare (z,pos,batch) via hooks; do not set extras yet:
        prep: PreparedBatch = self.hooks.prepare_inputs(x, edge_index, **kwargs)

        # Clusters (use provided edges if any; else 2-NN on coordinates):
        edges_np = kwargs.pop("edges", None)
        clusters = self._build_clusters(prep.z, prep.pos, edges_np)
        self._cached_clusters = clusters  # cache for forward()

        # Backbone edges and extras:
        mp_edge_index = self.hooks.make_edge_index(model, prep)
        prep = prep.with_(
            edge_index=mp_edge_index,
            extras=self.hooks.edge_attr_and_extras(model, prep),
        )

        # 1) capture *unmasked* node features at requested feature_point
        ones_mask = torch.ones(
            prep.z.size(0), 1, device=prep.z.device, dtype=torch.float32
        )
        _, captures = self.hooks.backbone_step(
            model, prep, ones_mask, where=self.feature_point
        )
        feats = self._pick_feats(captures)  # [N, H]

        # 2) pool to cluster embeddings [C,H]
        T = self._cluster_transform(
            clusters, num_nodes=feats.size(0), device=feats.device
        )  # [C,N]
        cluster_emb = (
            T @ feats
        )  # sum-pool; use mean at inference for stability if you like

        # 3) lazy-create MLP & optimizer now that H is known
        if self.mlp is None:
            in_dim = cluster_emb.size(1)
            self.mlp = Sequential(
                Linear(in_dim, self.hidden), ReLU(), Linear(self.hidden, 1)
            ).to(feats.device)
            self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)

        assert self.mlp is not None and self.optimizer is not None
        self.optimizer.zero_grad()

        # 4) predict cluster logits -> Concrete sample -> sigmoid -> atom mask
        temperature = self._get_temperature(epoch)
        logits_c = self.mlp(cluster_emb)  # [C,1]
        sample = self._concrete_sample(logits_c, temperature)  # [C,1] (logits w/ noise)
        mask_c = torch.sigmoid(sample)  # [C,1] (0..1)
        node_mask_atom = (mask_c.T @ T).T  # [N,1]

        # 5) masked forward & loss
        raw_out, _ = self.hooks.backbone_step(
            model, prep, node_mask_atom, where=self.mask_point
        )
        out = self.hooks.readout(model, raw_out, prep).view(-1)

        y_hat, y = out, target
        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss = self._loss(y_hat, y, mask_c)
        loss.backward()
        self.optimizer.step()

        self._curr_epoch = epoch
        clear_masks(model)  # just in case

        return float(loss.detach().cpu())

    # --------------------------------------------------------------------- #
    # Inference
    # --------------------------------------------------------------------- #
    def forward(
        self,
        model: nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if self._curr_epoch < self.epochs - 1:
            raise ValueError(
                f"'{self.__class__.__name__}' not fully trained: {self._curr_epoch + 1}/{self.epochs} epochs."
            )

        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(
                f"Heterogeneous graphs not supported in '{self.__class__.__name__}'"
            )

        model.eval()
        prep: PreparedBatch = self.hooks.prepare_inputs(x, edge_index, **kwargs)

        # Rebuild clusters for this input (or use cached if you prefer strict transductive eval):
        edges_np = kwargs.pop("edges", None)
        clusters = self._build_clusters(prep.z, prep.pos, edges_np)
        T = self._cluster_transform(
            clusters, num_nodes=prep.z.size(0), device=prep.z.device
        )

        # Get unmasked features at requested feature point:
        mp_edge_index = self.hooks.make_edge_index(model, prep)
        prep = prep.with_(
            edge_index=mp_edge_index,
            extras=self.hooks.edge_attr_and_extras(model, prep),
        )
        ones_mask = torch.ones(
            prep.z.size(0), 1, device=prep.z.device, dtype=torch.float32
        )
        _, captures = self.hooks.backbone_step(
            model, prep, ones_mask, where=self.feature_point
        )
        feats = self._pick_feats(captures)  # [N,H]

        # Predict mask deterministically:
        assert self.mlp is not None
        cluster_emb = (T @ feats) / (
            T.sum(dim=1, keepdim=True).clamp(min=1.0)
        )  # mean-pool at inference
        logits_c = self.mlp(cluster_emb)  # [C,1]
        mask_c = torch.sigmoid(logits_c)  # [C,1]
        node_mask = (mask_c.T @ T).T  # [N,1]

        clear_masks(model)

        return Explanation(
            node_mask=node_mask,  # atom-level mask [N,1]
            motif_mask=mask_c,  # cluster-level probs [C,1]
            clusters=clusters,  # list of clusters
        )

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _build_clusters(self, z: Tensor, pos: Tensor, edges_np=None) -> List[List[int]]:
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

    def _cluster_transform(
        self, clusters: List[List[int]], num_nodes: int, device
    ) -> Tensor:
        """
        Build [C,N] transform. If an atom is in multiple clusters, weight 1/deg.
        """
        T = torch.zeros(len(clusters), num_nodes, dtype=torch.float32, device=device)
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

    def _pick_feats(self, captures: dict) -> Tensor:
        """
        Select which node features to pool into cluster embeddings.
        """
        if self.feature_point == MaskPoint.EMBED:
            return captures["embed"]
        elif self.feature_point == MaskPoint.PRE_AGG:
            return captures["pre_agg"]
        else:
            raise ValueError(f"Unsupported feature_point: {self.feature_point}")

    # --------------------------------------------------------------------- #
    # Loss & sampling
    # --------------------------------------------------------------------- #
    def _get_temperature(self, epoch: int) -> float:
        t0, t1 = self.coeffs["temp"]
        return t0 * pow(t1 / t0, epoch / max(1, self.epochs))

    def _concrete_sample(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        bias = self.coeffs["bias"]
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(y_hat, y)

    def _loss(self, y_hat: Tensor, y: Tensor, cluster_mask: Tensor) -> Tensor:
        # Data-fit:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            raise AssertionError(f"Unsupported model mode: {self.model_config.mode}")

        # Regularization over cluster mask:
        m = torch.sigmoid(cluster_mask)
        size_loss = m.sum() * self.coeffs["edge_size"]
        m_smooth = 0.99 * m + 0.005
        ent = -m_smooth * m_smooth.log() - (1 - m_smooth) * (1 - m_smooth).log()
        ent_loss = ent.mean() * self.coeffs["edge_ent"]
        return loss + size_loss + ent_loss
