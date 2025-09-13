import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer

import numpy as np
from three_d_graphx.schnet.gnnexplainer.gnn_explainer import (
    GNNExplainer,
    schnet,
    datasets,
)

train_dataset, val_dataset, test_dataset = datasets
schnet.eval()

explainer = Explainer(
    model=schnet,
    algorithm=GNNExplainer(epochs=500),
    explanation_type='model',
    node_mask_type= 'object',
    edge_mask_type= None ,
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
    #threshold_config=dict(threshold_type ='topk', value = 7),
)


ass_loss = {}
explanation_list = [2,3,4,5,6,7,8,9]

for k in explanation_list:
    ass_loss[k] = []

for data in range(1024):
    explanation = explainer(test_dataset[data].z, test_dataset[data].pos, target=None)
    sorted_tensor, index = torch.sort(explanation.node_mask, dim = 0, descending= True)
    '''
    edges = explanation.edges
    selected_index3 = []
    for every_index in index.squeeze().numpy():
        if edges[0][every_index] not in selected_index3:
            selected_index3.append(edges[0][every_index].item())
        if edges[1][every_index] not in selected_index3:
            selected_index3.append(edges[1][every_index].item())
    '''

    for kk in explanation_list:
        with torch.no_grad():
            sorted_index = index.squeeze().numpy()[:kk]
            sorted_index.sort()
            pred1 = schnet(test_dataset[data].z[sorted_index],test_dataset[data].pos[sorted_index])
            loss = F.l1_loss(pred1, explanation.target)
        ass_loss[kk].append(loss.numpy())



print('result.......')
for kk in explanation_list:
    print('node number:',kk, ',mean->',np.mean(ass_loss[kk]), 'std->',np.std(ass_loss[kk]))

exit()
