from math import sqrt
from typing import Optional, Tuple, Union
import os
import os.path as osp
import argparse

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.datasets import QM9
from torch_geometric.nn import knn_graph
from torch_geometric.nn import DimeNet, DimeNetPlusPlus
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import scatter
from three_d_graphx.utils.threeD_motif import tree_decompose


parser = argparse.ArgumentParser()
parser.add_argument('--use_dimenet_plus_plus', action='store_true', default= True)
args = parser.parse_args()

dimnet = DimeNetPlusPlus if args.use_dimenet_plus_plus else DimeNet

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)
idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
dataset.data.y = dataset.data.y[:, idx]

device = torch.device('cpu')
target_attr = int(os.getenv('THREE_D_GRAPHX_TARGET_ATTR', '10'))
dimnet, datasets = dimnet.from_qm9_pretrained(path, dataset, target_attr)
train_dataset, val_dataset, test_dataset = datasets


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

        z = x
        pos = edge_index
        edge_list = knn_graph(edge_index, 2).numpy()
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
        #for (cluster, (_,rotable)) in zip(all_clusters, rotables.items()):
        #    if not rotable:
        #        clusters.append(cluster)

        clusters = all_clusters
        kwargs['clusters'] = clusters
        self._initialize_masks(x, edge_index,  **kwargs)
        edge_index = radius_graph(pos, r = dimnet.cutoff, batch=batch,
                                  max_num_neighbors=dimnet.max_num_neighbors)

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

        node_mask = self.node_mask.sigmoid()
        node_mask = torch.mm(node_mask.t(),transform_matrix).t()
        
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            from torch_geometric.nn.models.dimenet import triplets
            i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

            # Calculate angles.
            if isinstance(dimnet, DimeNetPlusPlus):
                pos_jk, pos_ij = pos[idx_j] - pos[idx_k], pos[idx_i] - pos[idx_j]
                a = (pos_ij * pos_jk).sum(dim=-1)
                b = torch.cross(pos_ij, pos_jk).norm(dim=-1)
            elif isinstance(dimnet, DimeNet):
                pos_ji, pos_ki = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_i]
                a = (pos_ji * pos_ki).sum(dim=-1)
                b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
            angle = torch.atan2(b, a)

            rbf = dimnet.rbf(dist)
            sbf = dimnet.sbf(dist, angle, idx_kj)

            # Embedding block.
            x = dimnet.emb(z, rbf, i, j)

            P = dimnet.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
            P = P * node_mask

            # Interaction blocks

            for interaction_block, output_block in zip(dimnet.interaction_blocks,
                                                   dimnet.output_blocks[1:]):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                P = P + output_block(x, rbf, i, num_nodes=pos.size(0))

            if batch is None:
                out = P.sum(dim=0)
            else:
                out = scatter(P, batch, dim=0, reduce='sum')
            #print('out is:', out, 'target is:', target)
            out = out.view(-1)
            y_hat, y = out, target

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y)
            #print('loss is =>', loss)

            loss.backward(retain_graph=True)
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if epoch == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if epoch == 0 and self.edge_mask is not None:
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
            
        device = x.device
        
        (N,), E = x.size(), edge_index.size(1)
        N = len(clusters) if clusters else 0
        F = 1
        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:

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


