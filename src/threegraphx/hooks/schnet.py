from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.models.schnet import SchNet

from .base import BackboneHooks, PreparedBatch, MaskPoint, normalize_node_mask


class SchNetHooks(BackboneHooks):
    """Hooks for SchNet with mask at EMBED or PRE_AGG."""

    def prepare_inputs(self, x: Tensor, edge_index: Tensor, **kw) -> PreparedBatch:
        # Convention: callers often pass x=z and edge_index=pos
        z = kw.get("z", x)
        pos = kw.get("pos", edge_index)
        batch = kw.get("batch", None)
        return PreparedBatch(z=z, pos=pos, batch=batch)

    def make_edge_index(self, model: SchNet, prep: PreparedBatch) -> Tensor:
        batch = prep.batch if prep.batch is not None else torch.zeros_like(prep.z)
        eidx, _ = model.interaction_graph(prep.pos, batch)
        return eidx

    def edge_attr_and_extras(self, model: SchNet, prep: PreparedBatch) -> Dict:
        batch = prep.batch if prep.batch is not None else torch.zeros_like(prep.z)
        eidx, edge_weight = model.interaction_graph(prep.pos, batch)
        edge_attr = model.distance_expansion(edge_weight)
        return {
            "eidx": eidx,
            "edge_weight": edge_weight,
            "edge_attr": edge_attr,
            "batch": batch,
        }

    def backbone_step(
        self,
        model: SchNet,
        prep: PreparedBatch,
        node_mask_atom: Tensor,
        *,
        where: MaskPoint,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        ex = prep.extras
        batch = ex["batch"]
        N = prep.z.size(0)
        m = normalize_node_mask(node_mask_atom, N)  # [N,1], float32

        # Earliest node embedding:
        h = model.embedding(prep.z)  # [N, H]
        embed_no_mask = h  # capture
        if where == MaskPoint.EMBED:
            h = h * m  # broadcast [N,1] over [N,H]

        # Edges/edge features:
        eidx, ew, eattr = ex["eidx"], ex["edge_weight"], ex["edge_attr"]

        # Interaction stack:
        for interaction in model.interactions:
            h = h + interaction(h, eidx, ew, eattr)

        # Projection:
        h = model.lin1(h)
        h = model.act(h)
        h = model.lin2(h)  # [N, D]
        pre_agg_no_mask = h  # capture

        if where == MaskPoint.PRE_AGG:
            h = h * m

        # Dipole / stats / atomref adjustments before readout:
        out = h
        if model.dipole:
            mass = model.atomic_mass[prep.z].view(-1, 1)
            M = model.sum_aggr(mass, batch, dim=0)
            c = model.sum_aggr(mass * prep.pos, batch, dim=0) / M
            out = out * (prep.pos - c.index_select(0, batch))
        if (not model.dipole) and (model.mean is not None) and (model.std is not None):
            out = out * model.std + model.mean
        if (not model.dipole) and (model.atomref is not None):
            out = out + model.atomref(prep.z)

        captures = {"embed": embed_no_mask, "pre_agg": pre_agg_no_mask}
        return out, captures

    def readout(self, model: SchNet, raw_out: Tensor, prep: PreparedBatch) -> Tensor:
        out = model.readout(raw_out, prep.extras["batch"], dim=0)
        if model.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)
        if model.scale is not None:
            out = model.scale * out
        return out
