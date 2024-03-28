#Use to generate motif，and calculate frequency,internal structure
import os.path as osp
import argparse
import itertools
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet
from torch_geometric.nn import knn_graph

from utils import node_label
from threeD_motif import tree_decompose

target = 0
dim = 64


parser = argparse.ArgumentParser()
parser.add_argument('--cutoff', type=float, default=10.0,
                    help='Cutoff distance for interatomic interactions')
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
dataset = QM9(path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_attr = 0
schnet, datasets = SchNet.from_qm9_pretrained(path, dataset, target_attr)
train_dataset, val_dataset, test_dataset = datasets
schnet = schnet.to(device)


#Brutal force to find the ground truth
def ground_truth(graph, motif_freq_dict, K_cluster = 3):
    z = graph.z
    edge_index = knn_graph(graph.pos, 1)
    
    #edge_index = graph.edge_index
    pos = graph.pos
    y =  graph.y[:,target_attr].double()
    full_pred = schnet(z, pos)[0]
    loss = F.l1_loss(full_pred, y)
    
    edges = set()
    atom_names = {}
    for (idx,atom) in enumerate(z):
        atom_names[idx] = node_label[atom.item()]
    for (start_idx, end_idx) in zip(edge_index[0].numpy(), edge_index[1].numpy()):
        if start_idx < end_idx:
            edges.add((start_idx, end_idx))
        else:
            edges.add((end_idx, start_idx))
    edges = list(edges)
    edges.sort(key=lambda x:(x[0], x[1]))
    clusters, rotable = tree_decompose(atom_names, edges)

    edges_index ={i: edge for (i, edge) in enumerate(edges)}
    reverse_edge_index ={edge : i for (i, edge) in enumerate(edges)}

    motif_length = len(clusters)
    
    K_cluster = K_cluster if K_cluster < motif_length else motif_length

    best_result =[]
    best_pred = None
    best_error = None
    loss_dif = np.inf
    for i in range(1, K_cluster+1):
        for j in combinations(range(motif_length), i): #motif:[1,2,3]
            selected_atoms = []
            selected_edges =[]
            selected_rotable = []
            atom_position_dict ={}
            motif_freq = {}
            for select_motif in j:
                motif_list = list()
                for atom in clusters[select_motif]:
                    selected_atoms.append(atom)
                    motif_list.append(node_label[z[atom].item()])
                if rotable[select_motif]:
                    selected_rotable.append(j)
                motif_list.sort()
                motif_string = ''.join(motif_list)
                motif_freq[motif_string] = motif_freq_dict.get(motif_string,0)
            #selected_atoms = list(set(selected_atoms))
            for (i,atom) in enumerate(selected_atoms):
                atom_position_dict[atom] = i

            selected_atoms = list(set(selected_atoms))
            selected_atoms.sort()
            pred2 = schnet(z[selected_atoms], pos[selected_atoms])[0]
            error2 = F.l1_loss(pred2, y)
            motif_loss = 0
            for (motif_name, motif_feq_num) in motif_freq.items():
                motif_loss +=motif_feq_num
            alpha,beta,gamma = 0.8,1,1
            regression_error = abs(error2 - loss)
            abs_loss_diff = regression_error*alpha + beta* len(selected_atoms)/len(z) + gamma *motif_loss
            if abs_loss_diff < loss_dif and len(selected_atoms) <= 5:
                best_pred = pred2
                best_error = error2
                loss_dif = abs_loss_diff
                best_result = selected_atoms
    
    
#use to generate motif, according atom number, mae
graph_motif_dict ={}
'''
{
    1:{  #graph_number
        'CH':{ #motif
            (0,4):[      #(atom:number) pair
                rotable,pred, target, mae #对应的信息
            ];
            (1,5):[      #(atom:number) pair
                rotable,pred, target, mae 
            ];  
        };
    };
}
'''

motifs_total =0
bonds_total =0

motif_2_graph_dict ={}
'''
{
    'CH':{
        1:{  #graph_number
            (0,4):[      #(atom:number) pair
                rotable,pred, target, mae #对应的信息
            ];
            (1,5):[      #(atom:number) pair
                rotable, pred, target, mae
            ]  
        };
    };
}

'''

total_node = 0
total_edge = 0
def motif_vocab(graph_number, graph):
    graph_motif_dict[graph_number] = list()
    z = graph.z
    pos = graph.pos
    edge_index = knn_graph(graph.pos, 1)
    adj = edge_index
    global total_edge, total_node
    total_node += len(z)
    total_edge += len(edge_index[0])
    label = graph.y[:,target_attr]
    edges = set()
    atom_names = {}
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
   
    global motifs_total,bonds_total
    motifs_total += int(len(clusters) - sum(rotable.values()))
    bonds_total += int(sum(rotable.values()))

    for i, c in enumerate(clusters):
        motif_string_list = list()
        list(c).sort()
        with torch.no_grad():
            pred1 = schnet(z[c], pos[c])[0]
            loss = F.l1_loss(pred1, label)
        
        for atom_seqence_number in c:
            motif_string_list.append(atom_names[atom_seqence_number])
        motif_string_list.sort()
        motif_string = ''.join(motif_string_list)

        #graph_motif_dict[graph_number][motif_string] = graph_motif_dict[graph_number].get(motif_string, {})
        #graph_motif_dict[graph_number][motif_string][tuple(c)] = [rotable[i],pred1, label,loss]

        motif_2_graph_dict[motif_string] = motif_2_graph_dict.get(motif_string, {})
        motif_2_graph_dict[motif_string][graph_number] =motif_2_graph_dict[motif_string].get(graph_number,{})
        #motif_2_graph_dict[motif_string][graph_number][tuple(c)] = [rotable[i],pred1.half(), label.half(),loss.half()]
        motif_2_graph_dict[motif_string][graph_number][tuple(c)] = [rotable[i],pred1.half(),loss.half().numpy()]

        

def motif_vocab_freq(motif_2_graph_dict):
    motif_freq = {}
    total_atom = 0
    motif_statis ={}

    number_of_graph_per_motif = dict()
    from collections import defaultdict
    number_of_clusters_per_motif = defaultdict(int)
    scores_per_motif = defaultdict(float)
    number_of_rotable_per_motif = defaultdict(int)
    for (motif_string, details) in motif_2_graph_dict.items():
        number_of_graph_per_motif[motif_string] = len(details)
        scores_per_motif[motif_string] = list()
        for (graph_id, cluster) in details.items():
            number_of_clusters_per_motif[motif_string] += len(cluster)
            total_atom += len(cluster)
            for(atom_tuple, result_list) in cluster.items():
                scores_per_motif[motif_string].append(result_list[2])
                
                number_of_rotable_per_motif[motif_string] += result_list[0]
    
    number_of_clusters_per_motif_by_order = sorted(number_of_clusters_per_motif.items(), key = lambda x : x[1], reverse = True)    
    #motif_dict_by_order = sorted(motif_dict.items(), key = lambda x : x[1], reverse = True)
    for(atom, atom_number) in number_of_clusters_per_motif_by_order:
        atom_freq = float(atom_number) / float(total_atom)
        motif_freq[atom] = atom_freq

    for(motif_string, loss_list) in scores_per_motif.items():
        means = np.mean(loss_list)
        stds = np.std(loss_list)
        motif_statis[motif_string] =[means,stds]
        #scores_per_motif[motif_string] = total_loss /(float)(number_of_clusters_per_motif[motif_string]) 
    
    for(motif_string, rotable) in number_of_rotable_per_motif.items():
        number_of_rotable_per_motif[motif_string] = float(rotable) /(float)(number_of_clusters_per_motif[motif_string])
    

    
    return motif_freq,motif_statis, number_of_rotable_per_motif, number_of_graph_per_motif


if __name__ =='__main__':
    
    #use to mining all the motif
    import pickle
    for (i,graph) in enumerate(train_dataset[:10000]):
        motif_vocab(i, graph)

    motif_freq,motif_statis, number_of_rotable_per_motif, number_of_graph_per_motif = motif_vocab_freq(motif_2_graph_dict)

    print('---------------')
    print(motif_freq)
    average_length =0.0
    for motif_string, freq  in motif_freq.items():
        average_length+=len(motif_string) *freq
    print('average_length', average_length)
    print('---------------')

    for motif_string, stas in motif_statis.items():
        print('motif statistics', motif_string, 'means', stas[0], 'std', stas[1])

    print('---------------')
    print(number_of_rotable_per_motif)
    print('---------------')
    print(number_of_graph_per_motif)

    print('motifs_total, bonds_total')
    print(motifs_total, bonds_total)
    
    print('total node:', total_node)
    print('total edge:', total_edge)
