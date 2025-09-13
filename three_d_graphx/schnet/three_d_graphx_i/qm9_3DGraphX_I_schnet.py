import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_graph
from torch_geometric.explain import Explainer

from three_d_graphx.schnet.three_d_graphx_i.graph_i import schnet, datasets, ThreeDGraphXI, target_attr


train_dataset, val_dataset, test_dataset = datasets

train_dataset = train_dataset[:2048]
val_dataset = val_dataset[:500]
test_dataset = test_dataset[:1024]

explainer = Explainer(
    model = schnet,
    algorithm= ThreeDGraphXI(epochs=50, lr=0.03),
    explanation_type='phenomenon',
    node_mask_type= None,
    edge_mask_type= 'object',
    model_config=dict(
        mode ='regression',
        task_level ='graph',
        return_type ='raw'
    ),
    #threshold_config=dict(threshold_type ='topk', value = 5),
)

if osp.exists('3dgraphx.pth'):
    checkpoint = torch.load('3dgraphx.pth')
    explainer.algorithm.load_state_dict(checkpoint['model_state_dict'])
    explainer.algorithm.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('加载 epoch {} 成功！'.format(start_epoch))
else:
    start_epoch = 0
    print('无保存模型，将从头开始训练！')


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 1,
    shuffle = True,
)


val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = 1,
    shuffle = True,
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 1,
    shuffle = True,
)


for epoch in range(start_epoch, 50):
    for data in train_loader:
        edge_index = knn_graph(data.pos,2 )
        loss = explainer.algorithm.train(
            epoch, schnet, data.z, data.pos, target=data.y[:,target_attr], edges= edge_index)
    print(epoch,'=>', loss)

    if (epoch+1) % 5 == 0:
        torch.save({'epoch':epoch,'model_state_dict':explainer.algorithm.state_dict(),
                        'optimizer_state_dict':explainer.algorithm.optimizer.state_dict(),}, '3dgraphx.pth')

# Generate the explanation for a particular graph:
#explanation = explainer(dataset[0].z, dataset[0].pos,target=dataset[0].y[:,target_attr].double())

ass_loss ={}
ass_loss2 = {}

explanation_list=[2,3,4,5,6,7,8,9]

for k in explanation_list:
    ass_loss[k] = []
    ass_loss2[k] = []

for i, data in enumerate(test_dataset[0:1024], start=0):
    edge_index = knn_graph(data.pos,2 )
    explanation = explainer(data.z, data.pos,target= data.y[:,target_attr].double(), edges = edge_index)
    pred3 = schnet(data.z, data.pos)
    sorted_tensor, index = torch.sort(explanation.node_mask, dim = 0, descending= True)
    sorted_tensor3, index3 = torch.sort(explanation.motif_mask, dim = 0, descending= True)
    selected_index3 = []
    for every_index in index3:  
        for every_atom in explanation.clusters[every_index]:
            if every_atom not in selected_index3:
                selected_index3.append(every_atom)


    ass_loss_dict = {}
    for k in explanation_list:
        with torch.no_grad():
            sorted_index = index.squeeze().numpy()[:k]
            sorted_index.sort()
            if i <= 30: print('graph:',i, 'k:', k,'kk->',sorted_index)
            pred1 = schnet(data.z[sorted_index], data.pos[sorted_index])[0]
            loss = F.l1_loss(pred1, explanation.target)

            sorted_index3= selected_index3[:k]
            sorted_index3.sort()
            if i <= 30: print('graph:',i, 'k:', k, 'kk3->',sorted_index3)
            pred3 = schnet(data.z[sorted_index3], data.pos[sorted_index3])[0]
            loss3 = F.l1_loss(pred3, explanation.target)
        #print(sorted_index, pred1, graph.y[:,0],loss)

        ass_loss[k].append(loss.numpy())
        ass_loss2[k].append(loss3.numpy())
        ass_loss_dict[k]=[sorted_index,pred1, data.y[:,0],loss.numpy()]

for k in explanation_list:
    print('k:',k,'-------->',np.mean(ass_loss[k]), np.std(ass_loss[k]))
    print('k:',k,'-------->',np.mean(ass_loss2[k]), np.std(ass_loss2[k]))
