import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer

from three_d_graphx.schnet.three_d_graphx_t.graph_t import ThreeGraphXT, schnet, datasets

train_dataset, val_dataset, test_dataset = datasets

explainer = Explainer(
    model=schnet,
    algorithm=ThreeGraphXT(epochs=500),
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
ass_loss2 = {}
explanation_list = [2,3,4,5,6,7,8,9]

for k in explanation_list:
    ass_loss[k] = []
    ass_loss2[k] = []


for data in test_dataset[:1024]:
    edge_index = knn_graph(data.pos, 2).numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    edge_losses = {}
    selected_edges =[]
    for edge in edges:
        edge_list = list(edge)
        edge_list.sort()
        a, b=edge_list
        with torch.no_grad():
            edge_pred = schnet(data.z[edge_list], data.pos[edge_list])[0]
            edge_loss = F.l1_loss(edge_pred, data.y[:,0]).numpy()

            if edge_loss < np.array(5.0):
                edge_losses[tuple(edge_list)] = edge_loss
    
    a = sorted(edge_losses.items(), key = lambda x: x[1])[:len(data.z)*2]
    edge_index = [first for first, second in a]
    
    explanation = explainer(data.z, data.pos, target=None, edges= edge_index)
    sorted_tensor, index = torch.sort(explanation.mask1, dim = 0, descending= True)
    sorted_tensor3, index3 = torch.sort(explanation.node_mask, dim = 0, descending= True)
    selected_index3 = []
    for every_index in index3:
        for every_atom in explanation.cluster[every_index]:
            if every_atom not in selected_index3:
                selected_index3.append(every_atom)
    

    for kk in explanation_list:
        with torch.no_grad():
            sorted_index = index.squeeze().numpy()[:kk]
            sorted_index.sort()
           
            pred1 = schnet(data.z[sorted_index], data.pos[sorted_index])
            loss = F.l1_loss(pred1, explanation.target)
            

            sorted_index3 = selected_index3[:kk]
            sorted_index3.sort()
            pred3 = schnet(data.z[sorted_index3], data.pos[sorted_index3])
            loss3 = F.l1_loss(pred3, explanation.target)

                

        ass_loss[kk].append(loss.numpy())
        ass_loss2[kk].append(loss3.numpy())
    
    
            


print('result.......')
for kk in explanation_list:
    print('node number:',kk, ',mean->',np.mean(ass_loss[kk]), 'std->',np.std(ass_loss[kk]))
    print('----node number-----:',kk, ',mean->',np.mean(ass_loss2[kk]), 'std->',np.std(ass_loss2[kk]))

exit()
