from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional, Dict, Tuple

import torch
from torch import Tensor


class MaskPoint(str, Enum):
    """Where to apply the node mask inside the backbone forward."""

    EMBED = "embed"  # earliest node embeddings
    PRE_AGG = "pre_agg"  # right before graph-level readout/aggregation


@dataclass(frozen=True)
class PreparedBatch:
    """Backbone-agnostic inputs prepared for a single masked forward pass."""

    z: Tensor
    pos: Tensor
    batch: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "PreparedBatch":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        return replace(
            self,
            z=self.z.to(device),
            pos=self.pos.to(device),
            batch=None if self.batch is None else self.batch.to(device),
            edge_index=None if self.edge_index is None else self.edge_index.to(device),
            extras={k: mv(v) for k, v in self.extras.items()},
        )

    def with_(self, **kwargs) -> "PreparedBatch":
        return replace(self, **kwargs)


def normalize_node_mask(mask: Tensor, num_nodes: int) -> Tensor:
    """Accept [N] or [N,1] and return float32 [N,1]; raise on mismatch."""
    if mask.dim() == 1 and mask.size(0) == num_nodes:
        mask = mask.unsqueeze(1)
    if mask.dim() != 2 or mask.size(0) != num_nodes or mask.size(1) != 1:
        raise ValueError(
            f"node_mask must be shape [N] or [N,1]; got {tuple(mask.shape)} for N={num_nodes}"
        )
    return mask.to(dtype=torch.float32)


class BackboneHooks(ABC):
    """Bridge implementor interface for different backbones (SchNet, DimeNet, ...)."""

    @abstractmethod
    def prepare_inputs(self, x: Tensor, edge_index: Tensor, **kw) -> PreparedBatch:
        """Map caller's (x, edge_index, **kw) into a PreparedBatch (z,pos,batch,...)."""
        ...

    @abstractmethod
    def make_edge_index(self, model, prep: PreparedBatch) -> Tensor:
        """Create the backbone's message-passing edges for this graph."""
        ...

    @abstractmethod
    def edge_attr_and_extras(self, model, prep: PreparedBatch) -> Dict[str, Any]:
        """Compute/cache backbone-specific edge features (rbf/sbf/weights/etc.)."""
        ...

    @abstractmethod
    def backbone_step(
        self,
        model,
        prep: PreparedBatch,
        node_mask_atom: Tensor,
        *,
        where: MaskPoint,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run one masked forward up to (but not including) graph readout.

        Returns:
            raw_out: node-level tensor ready for readout
            captures: optional intermediates, e.g. {"embed": Tensor, "pre_agg": Tensor}
        """
        ...

    @abstractmethod
    def readout(self, model, raw_out: Tensor, prep: PreparedBatch) -> Tensor:
        """Turn node-level raw_out into graph-level prediction tensor."""
        ...
