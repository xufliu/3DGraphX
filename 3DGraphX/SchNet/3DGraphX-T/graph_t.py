import os.path as osp
from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9
from torch_geometric.nn import knn_graph
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel

from ...Utils.threeD_motif import tree_decompose

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
target_attr = 10
schnet, datasets = SchNet.from_qm9_pretrained(path, dataset, target_attr)


class ThreeGraphXT(ExplainerAlgorithm):

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

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
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        
        self._train(model, x, edge_index, target=target, index=index, **kwargs)

        node_mask = self._post_process_mask(
            #self.node_mask,
            self.mask2,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)
        return Explanation(node_mask=self.node_mask, cluster=self.cluster, mask1= self.mask, mask2 = node_mask)

    def supports(self) -> bool:
        return True

    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        
        if 'edges' in kwargs:
            edge_list = kwargs['edges']
            del kwargs['edges']

        pos = edge_index
        edge_list = knn_graph(pos, 2).numpy()
        batch = kwargs['batch'] if 'batch' in kwargs else None
        batch = torch.zeros_like(x) if batch is None else batch
        
        parameters = []
        edges =set()
        
        for (start_idx, end_idx) in zip(edge_list[0], edge_list[1]):
            if start_idx < end_idx:
                edges.add((start_idx, end_idx))
            else:
                edges.add((end_idx, start_idx))
        
        edges = list(edges)
        edges.sort(key=lambda a:(a[0], a[1]))
        
        all_clusters, rotables = tree_decompose(x, edges)
        clusters = []
        for (cluster, (_,rotable)) in zip(all_clusters, rotables.items()):
            if not rotable:
                clusters.append(cluster)

        clusters = all_clusters
        kwargs['clusters'] = clusters
        self._initialize_masks(x, edge_index,  **kwargs)

        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index= edge_index, apply_sigmoid=True) 
            parameters.append(self.edge_mask)
        
        from collections import defaultdict
        atom_to_clusters = defaultdict(list)
        for (i,cluster) in enumerate(clusters):
            for atom in cluster:
                atom_to_clusters[atom].append(i)

        transform_matrix = torch.ones((len(clusters), len(x)), dtype = torch.float32)

        for (key, values) in atom_to_clusters.items():
            for value in values:
                transform_matrix[value, key] = 1/len(values)

        
        node_mask = torch.mm(self.node_mask.t(),transform_matrix).t()
        node_mask = node_mask.sigmoid()
        
        edge_index, edge_weight = schnet.interaction_graph(pos, batch)

        
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        
        for i in range(self.epochs):
            h = schnet.embedding(x)
            
            h = h *node_mask
            optimizer.zero_grad()
            
            #h = x if self.node_mask is None else x * torch.squeeze(self.node_mask.sigmoid())
            #y_hat, y = model(h, edge_index, **kwargs), target


            #src_attn = node_mask[edge_index[0]].squeeze()
            #dst_attn = node_mask[edge_index[1]].squeeze()
            
            #edge_weight = src_attn*dst_attn* edge_weight
            edge_attr = schnet.distance_expansion(edge_weight)
            
            #src_attn = node_mask[edge_index[0]]
            #dst_attn = node_mask[edge_index[1]]
            
            #edge_attr = dst_attn *src_attn* edge_attr

            for interaction in schnet.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = schnet.lin1(h)
            h = schnet.act(h)
            h = schnet.lin2(h)

            if schnet.dipole:
                # Get center of mass.
                mass = schnet.atomic_mass[x].view(-1, 1)
                M = schnet.sum_aggr(mass, batch, dim=0)
                c = schnet.sum_aggr(mass * pos, batch, dim=0) / M
                h = h * (pos - c.index_select(0, batch))

            if not schnet.dipole and schnet.mean is not None and schnet.std is not None:
                h = h * schnet.std + schnet.mean

            if not schnet.dipole and schnet.atomref is not None:
                h = h + schnet.atomref(x)

            out = schnet.readout(h, batch, dim=0)

            if schnet.dipole:
                out = torch.norm(out, dim=-1, keepdim=True)

            if schnet.scale is not None:
                out = schnet.scale * out

            y_hat, y = out, target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)*1000

            loss.backward(retain_graph=True)
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0
        self.mask = torch.mm(transform_matrix.t(), self.node_mask.sigmoid())
        self.mask2 = node_mask
        self.cluster = clusters

    def _initialize_masks(self, x: Tensor, edge_index: Tensor, **kwargs):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        if 'clusters' in kwargs:
            clusters = kwargs['clusters']
            del kwargs['clusters']
        
        #原始的
        device = x.device
        
        (N,), E = x.size(), edge_index.size(1)
        N = len(clusters) if clusters else N
        F = 1
        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            #first is the implementation of pyG，second is the implementation of LRI
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
            #std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            #self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            assert False

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(y_hat, y)    

    def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        '''
        size_loss = mask.mean()
        mask_ent_reg = -mask * (mask + 1e-10).log() - (1 - mask) * (1 - mask + 1e-10).log()
        mask_ent_loss = mask_ent_reg.mean()
        loss = loss + size_loss*0.01 + mask_ent_loss*0.1
        '''

        if self.hard_edge_mask is not None:
            assert self.edge_mask is not None
            m = self.edge_mask[self.hard_edge_mask].sigmoid()
            edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.hard_node_mask is not None:
            assert self.node_mask is not None
            m = self.node_mask[self.hard_node_mask].sigmoid()
            node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
            loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
            ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None

