//
// Normalized line endings to LF
//
#use to compare the internal structure by adding/delete node and perturb the locations.
import copy
import os
import os.path as osp
import argparse
from itertools import combinations, zip_longest
import pickle,copy


import torch
import torch.nn.functional as F

from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet
from torch_geometric.nn import knn_graph

import numpy as np

from three_d_graphx.utils.utils import node_label, Cartesian_2_Spherical, Spherical_2_Cartesian
from three_d_graphx.utils.threeD_motif import tree_decompose


parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_attr = int(os.getenv('THREE_D_GRAPHX_TARGET_ATTR', '0'))
schnet, datasets = SchNet.from_qm9_pretrained(path, dataset, target_attr)
train_dataset, val_dataset, test_dataset = datasets
schnet = schnet.to(device)
schnet.eval()


ce = torch.nn.CrossEntropyLoss(reduction='none')

target = 0
dim = 64



motif_2_graph_dict ={}

add_edge ={}

motifs_delete= []
bonds_delete = []

motifs_add= []
bonds_add = []

bond_length_noise = []
bond_theta_noise =[]
bond_phi_noise =[]

motif_length_noise = []
motif_theta_noise =[]
motif_phi_noise =[]

def node_perturbation(graph_number, graph, method ='delete'):

    z = graph.z
    pos = graph.pos
    edge_index = knn_graph(graph.pos, 1)
    #adj = graph.edge_index
    adj = edge_index
    label = graph.y[:,target_attr]
    edges = set()
    atom_names = {}
    graph_whole_pred = schnet(z, pos)[0]
    for (idx,atom) in enumerate(z):
        atom_names[idx] = node_label[atom.item()]
    for (start_idx, end_idx) in zip(adj[0].numpy(), adj[1].numpy()):
        if start_idx < end_idx:
            edges.add((start_idx, end_idx))
        else:
            edges.add((end_idx, start_idx))
    edges = list(edges)
    edges.sort(key=lambda x:(x[0], x[1]))

    clusters, rotable = tree_decompose(atom_names, edges)
    unrotable_clusters = []
    rotable_bonds = []
    for id,rotable_property in rotable.items():
        if rotable_property:
            rotable_bonds.append(clusters[id])
        else:
            unrotable_clusters.append(clusters[id])

    r_noise = np.random.uniform(-1, 1, 1)      #【0，∞）
    theta_noise = np.random.uniform(-np.pi/4,np.pi/4, 1)   #【0，π】
    phi_noise = np.random.uniform(-np.pi/2, np.pi/2, 1)   # [-π，π]

    for i, c in enumerate(clusters):
        c.sort()
        original_pred = schnet(z[c], pos[c])[0]

        motif_string_list = list()
        for atom_seqence_number in c:
            motif_string_list.append(atom_names[atom_seqence_number])
        motif_string_list.sort()
        motif_string = ''.join(motif_string_list)

        if method =='add':
            new_c = c.copy()
            list(new_c).sort()
            for j in range(len(c)):
                new_c=new_c[:j]+new_c[j+1:]
                pred1 = schnet(z[new_c], pos[new_c])[0]
                loss = F.l1_loss(pred1, original_pred)
                if not rotable[i]: 
                    motifs_delete.append(loss.detach().numpy())
                else:
                    bonds_delete.append(loss.detach().numpy())
        if method == 'delete':
            if rotable[i]:
                for atom in c:
                    for edge in zip_longest(list([atom]), list(range(len(z))), fillvalue =atom):
                        comb = set(c)
                        if list(edge) in unrotable_clusters:
                            comb.add(edge[1])
                            comb = list(comb)
                            pred1 = schnet(z[comb], pos[comb])[0]
                            loss = F.l1_loss(pred1, original_pred)
                            bonds_add.append(loss.detach().numpy())
            else:
                for atom in c:
                    for edge in zip_longest(list([atom]), list(range(len(z))), fillvalue =atom):
                        comb = set(c)
                        if list(edge) in rotable_bonds:
                            comb.add(edge[1])
                            comb = list(comb)
                            pred1 = schnet(z[comb], pos[comb])[0]
                            loss = F.l1_loss(pred1, original_pred)
                            motifs_add.append(loss.detach().numpy())
        if method == 'noise':
            if rotable[i]:
                pos_new = copy.deepcopy(pos)
                new_c = c.copy()
                relative_positions = Cartesian_2_Spherical(pos[new_c[0]], pos[new_c[1]])
                r, theta, phi=relative_positions[0]
                whole_position_except_c = pos_new
                whole_position_except_c = torch.cat(
                    (whole_position_except_c[:new_c[0]],whole_position_except_c[new_c[0] +1:new_c[1]], whole_position_except_c[new_c[1]+1:]), 
                    dim = 0)
                #del whole_position_except_c[new_c[0]]
                #del whole_position_except_c[new_c[1]]
                whole_position_except_c = whole_position_except_c - pos[new_c[1]]
                new_r = r + r_noise
                if new_r <= 0: continue
                new_c_1_pos = torch.from_numpy(Spherical_2_Cartesian(new_r, theta ,phi)).squeeze()
                new_c_1_pos = new_c_1_pos + pos[new_c[0]]
                whole_position_except_c_new = whole_position_except_c + new_c_1_pos
                whole_position_new = torch.cat(
                    (whole_position_except_c_new[:new_c[0]],
                     torch.unsqueeze(pos[new_c[0]], dim = 0),
                     whole_position_except_c_new[new_c[0]:new_c[1]],
                     torch.unsqueeze(new_c_1_pos, dim = 0),
                     whole_position_except_c_new[new_c[1]:]
                    ),
                    dim = 0
                ).float()
                #whole_position_except_c_new.insert(new_c[0], pos[new_c[0]])
                #whole_position_except_c_new.insert(new_c[0], new_c_1_pos)
                pos_new[new_c[1]] =new_c_1_pos
                #new_pos=torch.stack((pos[new_c[0]],new_c_1_pos)).float()
                pred1 = schnet(z, whole_position_new)[0]
                loss = F.l1_loss(pred1, graph_whole_pred)
                bond_length_noise.append(loss.detach().numpy())

                new_theta = theta + theta_noise
                if new_theta <= 0: new_theta += np.pi
                if new_theta > np.pi: new_theta -= np.pi
                new_c_2_pos = torch.from_numpy(Spherical_2_Cartesian(r, new_theta ,phi)).squeeze()
                new_c_2_pos = new_c_2_pos + pos[new_c[0]]
                whole_position_except_c_new = whole_position_except_c + new_c_2_pos
                whole_position_new = torch.cat(
                    (whole_position_except_c_new[:new_c[0]],
                     torch.unsqueeze(pos[new_c[0]], dim = 0),
                     whole_position_except_c_new[new_c[0]:new_c[1]],
                     torch.unsqueeze(new_c_2_pos, dim = 0),
                     whole_position_except_c_new[new_c[1]:]
                    ),
                    dim = 0
                ).float()
                pos_new[new_c[1]] =new_c_2_pos
                #new_pos=torch.stack((pos[new_c[0]],new_c_2_pos)).float()
                pred2 = schnet(z, whole_position_new)[0]
                loss = F.l1_loss(pred2, graph_whole_pred)
                bond_theta_noise.append(loss.detach().numpy())

                new_phi = phi + phi_noise
                if new_phi <= -np.pi: new_phi += np.pi
                if new_phi > np.pi: new_phi -= np.pi
                new_c_3_pos = torch.from_numpy(Spherical_2_Cartesian(r, theta ,new_phi)).squeeze()
                new_c_3_pos = new_c_3_pos + pos[new_c[0]]

                whole_position_except_c_new = whole_position_except_c + new_c_3_pos
                whole_position_new = torch.cat(
                    (whole_position_except_c_new[:new_c[0]],
                     torch.unsqueeze(pos[new_c[0]], dim = 0),
                     whole_position_except_c_new[new_c[0]:new_c[1]],
                     torch.unsqueeze(new_c_3_pos, dim = 0),
                     whole_position_except_c_new[new_c[1]:]
                    ),
                    dim = 0
                ).float()
                pos_new[new_c[1]] =new_c_3_pos
                #new_pos=torch.stack((pos[new_c[0]],new_c_3)).float()
                pred3 = schnet(z, whole_position_new)[0]
                loss = F.l1_loss(pred3, graph_whole_pred)
                bond_phi_noise.append(loss.detach().numpy())
            else:
                if len(c) == 2:
                    pos_new = copy.deepcopy(pos)
                    new_c = c.copy()
                    relative_positions = Cartesian_2_Spherical(pos[new_c[0]], pos[new_c[1]])
                    r, theta, phi = relative_positions[0]
                    whole_position_except_c = pos_new
                    whole_position_except_c = torch.cat(
                        (whole_position_except_c[:new_c[0]],whole_position_except_c[new_c[0] +1:new_c[1]], whole_position_except_c[new_c[1]+1:]), 
                        dim = 0)
                    whole_position_except_c = whole_position_except_c - pos[new_c[1]]
                    new_r = r + r_noise
                    if new_r <= 0: continue
                    new_c_1_pos = torch.from_numpy(Spherical_2_Cartesian(new_r, theta ,phi)).squeeze()
                    new_c_1_pos = new_c_1_pos + pos[new_c[0]]
                    whole_position_except_c_new = whole_position_except_c + new_c_1_pos
                    whole_position_new = torch.cat(
                        (whole_position_except_c_new[:new_c[0]],
                        torch.unsqueeze(pos[new_c[0]], dim = 0),
                        whole_position_except_c_new[new_c[0]:new_c[1]],
                        torch.unsqueeze(new_c_1_pos, dim = 0),
                        whole_position_except_c_new[new_c[1]:]
                        ),
                        dim = 0
                    ).float()
                    pos_new[new_c[1]] = new_c_1_pos
                    #new_pos=torch.stack((pos[new_c[0]],new_c_1)).float()
                    pred1 = schnet(z, whole_position_new)[0]
                    loss = F.l1_loss(pred1, graph_whole_pred)
                    motif_length_noise.append(loss.detach().numpy())

                    new_theta = theta + theta_noise
                    if new_theta <= 0: new_theta += np.pi
                    if new_theta > np.pi: new_theta -= np.pi
                    new_c_2_pos = torch.from_numpy(Spherical_2_Cartesian(r, new_theta ,phi)).squeeze()
                    new_c_2_pos = new_c_2_pos + pos[new_c[0]]
                    pos_new[new_c[1]] = new_c_2_pos
                    whole_position_except_c_new = whole_position_except_c + new_c_2_pos
                    whole_position_new = torch.cat(
                        (whole_position_except_c_new[:new_c[0]],
                        torch.unsqueeze(pos[new_c[0]], dim = 0),
                        whole_position_except_c_new[new_c[0]:new_c[1]],
                        torch.unsqueeze(new_c_2_pos, dim = 0),
                        whole_position_except_c_new[new_c[1]:]
                        ),
                        dim = 0
                    ).float()
                    #new_pos=torch.stack((pos[new_c[0]],new_c_2)).float()
                    pred2 = schnet(z, whole_position_new)[0]
                    loss = F.l1_loss(pred2, graph_whole_pred)
                    motif_theta_noise.append(loss.detach().numpy())

                    new_phi = phi + phi_noise
                    if new_phi <= -np.pi: new_phi += np.pi
                    if new_phi > np.pi: new_phi -= np.pi
                    new_c_3_pos = torch.from_numpy(Spherical_2_Cartesian(r, theta ,new_phi)).squeeze()
                    new_c_3_pos = new_c_3_pos + pos[new_c[0]]
                    whole_position_except_c_new = whole_position_except_c + new_c_3_pos
                    whole_position_new = torch.cat(
                        (whole_position_except_c_new[:new_c[0]],
                        torch.unsqueeze(pos[new_c[0]], dim = 0),
                        whole_position_except_c_new[new_c[0]:new_c[1]],
                        torch.unsqueeze(new_c_3_pos, dim = 0),
                        whole_position_except_c_new[new_c[1]:]
                        ),
                        dim = 0
                    ).float()
                    pos_new[new_c[1]] = new_c_3_pos
                    #new_pos=torch.stack((pos[new_c[0]],new_c_3)).float()
                    pred3 = schnet(z, whole_position_new)[0]
                    loss = F.l1_loss(pred3, graph_whole_pred)
                    motif_phi_noise.append(loss.detach().numpy())
                
                
                else:
                    
                    new_c = c.copy()
                    cal_common_atom ={}
                    for atom in new_c:
                        cal_common_atom[atom] = 0
                    for atom_combinations in combinations(new_c,2):
                        if atom_combinations in edges:
                            cal_common_atom[atom_combinations[0]] += 1
                            cal_common_atom[atom_combinations[1]] += 1
                    
                    #找key atom
                    appear = 0
                    key_atom = None
                    for (atom, appears) in cal_common_atom.items():
                        if appears > appear:
                            key_atom = atom
                            appear = appears
                    
                    rest_atoms = list(set(c) - set([key_atom]))
                    
                    relative_positions = Spherical_2_Cartesian(pos[key_atom], pos[rest_atoms])
                    new_position = []
                    new_position.append(pos[new_c[0]])
                    
                    for rest_atom, relative_position in zip(rest_atoms,relative_positions):
                        pos_new = copy.deepcopy(pos)
                        whole_position_except_c = pos_new
                        whole_position_except_c = torch.cat(
                        (whole_position_except_c[:key_atom],whole_position_except_c[key_atom +1:rest_atom], whole_position_except_c[rest_atom+1:]), 
                        dim = 0)
                        whole_position_except_c = whole_position_except_c - pos[rest_atom]
                        r, theta, phi =relative_position
                        new_r = r + r_noise
                        if new_r <= 0: continue
                        new_c_1_pos = torch.from_numpy(Spherical_2_Cartesian(new_r, theta ,phi)).squeeze()
                        new_c_1_pos = new_c_1_pos + pos[key_atom]
                        whole_position_except_c_new = whole_position_except_c + new_c_1_pos
                        whole_position_new = torch.cat(
                            (whole_position_except_c_new[:key_atom],
                            torch.unsqueeze(pos[key_atom], dim = 0),
                            whole_position_except_c_new[key_atom:rest_atom],
                            torch.unsqueeze(new_c_1_pos, dim = 0),
                            whole_position_except_c_new[rest_atom:]
                            ),
                            dim = 0
                        ).float()
                        new_position.append(new_c_1_pos)
                        pos_new[rest_atom] = new_c_1_pos
                        pred1 = schnet(z, whole_position_new)[0]
                        loss = F.l1_loss(pred1, graph_whole_pred)
                        motif_length_noise.append(loss.detach().numpy())


                    new_position = []
                    new_position.append(pos[new_c[0]])
                    for rest_atom,relative_position in zip(rest_atoms,relative_positions):
                        pos_new = copy.deepcopy(pos)
                        whole_position_except_c = pos_new
                        whole_position_except_c = torch.cat(
                        (whole_position_except_c[:key_atom],whole_position_except_c[key_atom +1:rest_atom], whole_position_except_c[rest_atom+1:]), 
                        dim = 0)
                        whole_position_except_c = whole_position_except_c - pos[rest_atom]
                        r, theta, phi = relative_position
                        new_theta = theta + theta_noise
                        if new_theta <= 0: new_theta += np.pi
                        if new_theta > np.pi: new_theta -= np.pi
                        new_c_2_pos = torch.from_numpy(Spherical_2_Cartesian(r, new_theta ,phi)).squeeze()
                        new_c_2_pos = new_c_2_pos + pos[key_atom]
                        pos_new[rest_atom] = new_c_2_pos
                        new_position.append(new_c_2_pos)
                        whole_position_except_c_new = whole_position_except_c + new_c_2_pos
                        whole_position_new = torch.cat(
                            (whole_position_except_c_new[:key_atom],
                            torch.unsqueeze(pos[key_atom], dim = 0),
                            whole_position_except_c_new[key_atom:rest_atom],
                            torch.unsqueeze(new_c_2_pos, dim = 0),
                            whole_position_except_c_new[rest_atom:]
                            ),
                            dim = 0
                        ).float()
                        pred2 = schnet(z, whole_position_new)[0]
                        loss = F.l1_loss(pred2, graph_whole_pred)
                        motif_theta_noise.append(loss.detach().numpy())


                    new_position = []
                    new_position.append(pos[new_c[0]])
                    for rest_atom,relative_position in zip(rest_atoms,relative_positions):
                        pos_new = copy.deepcopy(pos)
                        whole_position_except_c = pos_new
                        whole_position_except_c = torch.cat(
                        (whole_position_except_c[:key_atom],whole_position_except_c[key_atom +1:rest_atom], whole_position_except_c[rest_atom+1:]), 
                        dim = 0)
                        whole_position_except_c = whole_position_except_c - pos[rest_atom]
                        r, theta, phi = relative_position
                        new_phi = phi + phi_noise
                        if new_phi <= -np.pi: new_phi += np.pi
                        if new_phi > np.pi: new_phi -= np.pi
                        new_c_3_pos = torch.from_numpy(Cartesian_2_Spherical(r, theta ,new_phi)).squeeze()
                        new_c_3_pos = new_c_3_pos + pos[key_atom]
                        whole_position_except_c_new = whole_position_except_c + new_c_3_pos
                        whole_position_new = torch.cat(
                            (whole_position_except_c_new[:key_atom],
                            torch.unsqueeze(pos[key_atom], dim = 0),
                            whole_position_except_c_new[key_atom:rest_atom],
                            torch.unsqueeze(new_c_3_pos, dim = 0),
                            whole_position_except_c_new[rest_atom:]
                            ),
                            dim = 0
                        ).float()
                        pos_new[rest_atom] = new_c_3_pos
                        new_position.append(new_c_3_pos)
                        pred3 = schnet(z, whole_position_new)[0]
                        loss = F.l1_loss(pred3, graph_whole_pred)
                        motif_phi_noise.append(loss.detach().numpy())


if __name__ =='__main__':
    losses = []
    ass_loss_dict ={}
    for (i,graph) in enumerate(train_dataset[:10000]):
        node_perturbation(i, graph, method='noise')

    '''
    print('delete.................')
    print('it is motif delete')
    print(np.mean(motifs_delete), np.std(motifs_delete))
    print('it is bond delete')
    print(np.mean(bonds_delete), np.std(bonds_delete))

    print('add.................')
    print('it is motif add')
    print(np.mean(motifs_add), np.std(motifs_add))
    print('it is bond add')
    print(np.mean(bonds_add), np.std(bonds_add))
    '''

    print('noise.................')
    print('it is motif add')
    print(np.mean(motif_length_noise), np.std(motif_length_noise))
    print(np.mean(motif_theta_noise), np.std(motif_theta_noise))
    print(np.mean(motif_phi_noise), np.std(motif_phi_noise))
    print('it is bond add')
    print(np.mean(bond_length_noise), np.std(bond_length_noise))
    print(np.mean(bond_theta_noise), np.std(bond_theta_noise))
    print(np.mean(bond_phi_noise), np.std(bond_phi_noise))
