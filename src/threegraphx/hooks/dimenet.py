from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.dimenet import DimeNet, DimeNetPlusPlus, triplets
from torch_geometric.utils import scatter

from .base import BackboneHooks, PreparedBatch, MaskPoint, normalize_node_mask


class DimeNetHooks(BackboneHooks):
    """Hooks for DimeNet / DimeNet++ with mask at EMBED or PRE_AGG."""

    def prepare_inputs(self, x: Tensor, edge_index: Tensor, **kw) -> PreparedBatch:
        # Convention: callers often pass x=z and edge_index=pos
        z = kw.get("z", x)
        pos = kw.get("pos", edge_index)
        batch = kw.get("batch", None)
        return PreparedBatch(z=z, pos=pos, batch=batch)

    def make_edge_index(self, model: DimeNet, prep: PreparedBatch) -> Tensor:
        return radius_graph(
            prep.pos,
            r=model.cutoff,
            batch=prep.batch,
            max_num_neighbors=model.max_num_neighbors,
        )

    def edge_attr_and_extras(self, model: DimeNet, prep: PreparedBatch) -> Dict:
        eidx = self.make_edge_index(model, prep)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            eidx, num_nodes=prep.z.size(0)
        )
        dist = (prep.pos[i] - prep.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Angles:
        if isinstance(model, DimeNetPlusPlus):
            pos_jk = prep.pos[idx_j] - prep.pos[idx_k]
            pos_ij = prep.pos[idx_i] - prep.pos[idx_j]
            a = (pos_ij * pos_jk).sum(dim=-1)
            b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        else:  # DimeNet
            pos_ji = prep.pos[idx_j] - prep.pos[idx_i]
            pos_ki = prep.pos[idx_k] - prep.pos[idx_i]
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = model.rbf(dist)
        sbf = model.sbf(dist, angle, idx_kj)
        return {
            "eidx": eidx,
            "i": i,
            "rbf": rbf,
            "sbf": sbf,
            "idx_kj": idx_kj,
            "idx_ji": idx_ji,
        }

    def backbone_step(
        self,
        model: DimeNet,
        prep: PreparedBatch,
        node_mask_atom: Tensor,
        *,
        where: MaskPoint,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        ex = prep.extras
        N = prep.z.size(0)
        m = normalize_node_mask(node_mask_atom, N)  # [N,1]

        # Embedding + first node-level projection:
        x = model.emb(prep.z, ex["rbf"], ex["i"], ex["eidx"][1])
        P = model.output_blocks[0](
            x, ex["rbf"], ex["i"], num_nodes=prep.pos.size(0)
        )  # [N, H']
        embed_no_mask = P
        if where == MaskPoint.EMBED:
            P = P * m

        # Interaction/output blocks:
        for ib, ob in zip(model.interaction_blocks, model.output_blocks[1:]):
            x = ib(x, ex["rbf"], ex["sbf"], ex["idx_kj"], ex["idx_ji"])
            Q = ob(x, ex["rbf"], ex["i"], num_nodes=prep.pos.size(0))
            if where == MaskPoint.EMBED:
                Q = Q * m
            P = P + Q

        pre_agg_no_mask = P
        if where == MaskPoint.PRE_AGG:
            P = P * m

        captures = {"embed": embed_no_mask, "pre_agg": pre_agg_no_mask}
        return P, captures

    def readout(self, model: DimeNet, raw_out: Tensor, prep: PreparedBatch) -> Tensor:
        if prep.batch is None:
            return raw_out.sum(dim=0)
        return scatter(raw_out, prep.batch, dim=0, reduce="sum")
