import os.path as osp
import numpy as np


import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer


from lri_bern import LriBern, schnet, datasets, target_attr

train_dataset, val_dataset, test_dataset = datasets
train_dataset = train_dataset[:2048]
test_dataset = test_dataset[0:1024]

explainer = Explainer(
    model = schnet,
    algorithm= LriBern(epochs=100, lr=0.003),
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


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 128,
    shuffle = True,
)


test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 128,
    shuffle = False,
)


if osp.exists('lri_bern.pth'):
    checkpoint = torch.load('lri_bern.pth')
    explainer.algorithm.load_state_dict(checkpoint['model_state_dict'])
    explainer.algorithm.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print('loading epoch {} successfully'.format(start_epoch))
else:
    start_epoch = 0
    print('training from scratch')


for epoch in range(start_epoch, 100):
    train_loss = 0
    for data in train_loader:
        loss = explainer.algorithm.train(
            epoch, schnet, data.z, data.pos, target =data.y[:,target_attr],batch = data.batch)
        train_loss += loss * data.num_graphs
    print(epoch,'=>', train_loss /len(train_loader.dataset))
    
    if (epoch+1) % 5 == 0:
        torch.save({'epoch':epoch,'model_state_dict':explainer.algorithm.state_dict(),
                        'optimizer_state_dict':explainer.algorithm.optimizer.state_dict(),}, 'lri_bern.pth')
    

# Generate the explanation for a particular graph:
#explanation = explainer(dataset[0].z, dataset[0].pos,target=dataset[0].y[:,target_attr].double())

ass_loss ={}
ass_loss_dict = {}

explanation_list=[2,3,4,5,6,7,8,9]

for k in explanation_list:
    ass_loss[k] = []

for data in test_loader:
    explanation = explainer(data.z, data.pos, target=data.y[:,target_attr], batch = data.batch)
    for j in range(128):
        node_mask = explanation.node_mask[explanation.batch == j]
        sorted_tensor, index = torch.sort(node_mask, dim = 0, descending= True)
        ass_loss_dict = {}
        for kk in explanation_list:
            with torch.no_grad():
                sorted_index = index.squeeze().numpy()[:kk]
                sorted_index.sort()
                pred1 = schnet(data.z[sorted_index], data.pos[sorted_index])
                loss = F.l1_loss(pred1[0,0], explanation.target[j])

            ass_loss[kk].append(loss.numpy())

print('result.......')
for kk in explanation_list:
    print('node number:',kk, ',mean->',np.mean(ass_loss[kk]), 'std->',np.std(ass_loss[kk]))

exit()
