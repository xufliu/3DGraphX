import logging
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU, Sequential
import torch.nn.functional as F
from torch.nn import Dropout

from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear,BatchNorm
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings

from ...Utils.threeD_motif import tree_decompose

import os.path as osp
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_attr = 0
schnet, datasets = SchNet.from_qm9_pretrained(path, dataset, target_attr)
schnet = schnet.to(device)

class ThreeDGraphXI(ExplainerAlgorithm):

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.0,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 64),
            BatchNorm(64),
            ReLU(),
            #Dropout(0.2, inplace= False),
            #Linear(128, 64),
            #BatchNorm(64),
            #ReLU(),
            #Dropout(0.2, inplace= False),
            Linear(64, 1),
        )
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features of a
                homogeneous graph.
            edge_index (torch.Tensor): The input edge indices of a homogeneous
                graph.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model
                output to explain. Needs to be a single index.
                (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")


        z = x
        pos = edge_index
        origin_edge_index = edge_index
        if 'edges' in kwargs:
            edge_list = kwargs['edges']
            del kwargs['edges']

        edges = set()
        for (start_idx, end_idx) in zip(edge_list[0].numpy(), edge_list[1].numpy()):
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
        from collections import defaultdict
        atom_to_clusters = defaultdict(list)
        for (i,cluster) in enumerate(clusters):
            for atom in cluster:
                atom_to_clusters[atom].append(i)

        transform_matrix = torch.zeros((len(clusters), len(z)), dtype = torch.float32)
        for (key, values) in atom_to_clusters.items():
            for value in values:
                transform_matrix[value, key] = 1/len(values)

        emb = get_embeddings(model, z = z, pos = pos)[-1]
        cluster_emb = torch.ones((len(clusters), emb.shape[1])).type_as(emb)
        for (i,cluster) in enumerate(clusters):
            for atom in cluster:
                #cluster_emb[i,:] += (emb[atom, :] /len(cluster))
                cluster_emb[i,:] += (emb[atom, :] )

        batch = torch.zeros_like(z)
        
        edge_index, edge_weight = schnet.interaction_graph(pos, batch)
        

        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        logits = self.mlp(cluster_emb)

        node_mask = self._concrete_sample(logits, temperature).sigmoid()
        node_mask = torch.mm(node_mask.t(),transform_matrix).t()
        
        #src_attn = node_mask[edge_index[0]].squeeze()
        #dst_attn = node_mask[edge_index[1]].squeeze()
        #edge_weight = src_attn * dst_attn* edge_weight
        edge_attr = schnet.distance_expansion(edge_weight)

        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))
            edge_mask = edge_mask[hard_edge_mask]

        h = schnet.embedding(z)
        h = h * node_mask
        for interaction in schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = schnet.lin1(h)
        h = schnet.act(h)
        h = schnet.lin2(h)

        if schnet.dipole:
            # Get center of mass.
            mass = schnet.atomic_mass[z].view(-1, 1)
            M = schnet.sum_aggr(mass, batch, dim=0)
            c = schnet.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not schnet.dipole and schnet.mean is not None and schnet.std is not None:
            h = h * schnet.std + schnet.mean

        if not schnet.dipole and schnet.atomref is not None:
            h = h + schnet.atomref(z)

        out = schnet.readout(h, batch, dim=0)

        if schnet.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if schnet.scale is not None:
            out = schnet.scale * out

        y_hat = out[0]
        y = target.double()
        #y_hat, y = model(x, edge_index, **kwargs), target

        if index is not None:
            y_hat, y = y_hat[index], y[index]

        loss = self._loss(y_hat, y, node_mask)
        loss.backward()
        self.optimizer.step()

        clear_masks(model)
        self._curr_epoch = epoch

        return float(loss)

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

        if self._curr_epoch < self.epochs - 1:  # Safety check:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
                             f"trained (got {self._curr_epoch + 1} epochs "
                             f"from {self.epochs} epochs). Please first train "
                             f"the underlying explainer model by running "
                             f"`explainer.algorithm.train(...)`.")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))

        if 'edges' in kwargs:
            edge_list = kwargs['edges']
            del kwargs['edges']

        edges = set()
        for (start_idx, end_idx) in zip(edge_list[0].numpy(), edge_list[1].numpy()):
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

        #clusters = edges
        from collections import defaultdict
        atom_to_clusters = defaultdict(list)
        for (i,cluster) in enumerate(clusters):
            for atom in cluster:
                atom_to_clusters[atom].append(i)

        transform_matrix = torch.zeros((len(clusters), len(x)), dtype = torch.float32)
        for (key, values) in atom_to_clusters.items():
            for value in values:
                transform_matrix[value, key] = 1/len(values)

        
        
        emb = get_embeddings(model, x, edge_index, **kwargs)[-1]
        pos = edge_index
        batch = torch.zeros_like(x)

        cluster_emb = torch.zeros((len(clusters), emb.shape[1])).type_as(emb)
        for (i,cluster) in enumerate(clusters):
            for atom in cluster:
                cluster_emb[i,:] += (emb[atom, :] /len(cluster))
        
        edge_index, edge_weight = schnet.interaction_graph(pos, batch)
        edge_attr = schnet.distance_expansion(edge_weight)

        logits1 = self.mlp(cluster_emb)
        logits = torch.mm(logits1.t(),transform_matrix).t()

        node_mask = self._post_process_mask(logits, hard_edge_mask,
                                            apply_sigmoid=True)
        

        return Explanation(node_mask=node_mask, motif_mask = logits1.sigmoid(), clusters=clusters )

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return F.l1_loss(y_hat, y)

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss
