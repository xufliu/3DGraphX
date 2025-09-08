import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.explain import Explainer

from three_d_graphx.dimenet.three_d_graphx_t.graph_t_dimnet import ThreeGraphXT, datasets, dimnet

train_dataset, val_dataset, test_dataset = datasets


explainer = Explainer(
    model=dimnet,
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


for i,data in enumerate(test_dataset[:1024]):
    
    explanation = explainer(data.z, data.pos, target=None)
    sorted_tensor, index = torch.sort(explanation.mask2, dim = 0, descending= True)
    sorted_tensor2, index2 = torch.sort(explanation.mask1, dim = 0, descending= True)

    for kk in explanation_list:
        with torch.no_grad():
            sorted_index = index.squeeze().numpy()[:kk]
            sorted_index.sort()
            pred1 = dimnet(data.z[sorted_index], data.pos[sorted_index])
            loss = F.l1_loss(pred1, explanation.target)

            sorted_index2 = index2.squeeze().numpy()[:kk]
            sorted_index2.sort()
            pred2 = dimnet(data.z[sorted_index2], data.pos[sorted_index2])
            loss2 = F.l1_loss(pred2, explanation.target)      

        ass_loss[kk].append(loss.numpy())
        ass_loss2[kk].append(loss2.numpy())


print('result.......')
for kk in explanation_list:
    print('node number:',kk, ',mean->',np.mean(ass_loss[kk]), 'std->',np.std(ass_loss[kk]))
    print('----node number-----:',kk, ',mean->',np.mean(ass_loss2[kk]), 'std->',np.std(ass_loss2[kk]))
exit()
