node_label=['X','H','X','X','X','X','C','N','O','F','X','X','X','X']

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

def draw(G_copy, feat):
    G = G_copy.copy()
    labels ={}
    
    G.remove_nodes_from(list(nx.isolates(G)))
    
    pos = nx.spring_layout(G)      
    plt.rcParams['figure.figsize'] = (6,4)  
    nodes_protein_name_dict=dict()
    for G_node in G.nodes():
            feature_where_no_zero = torch.where(feat[0,G_node] == 1)[0]
            if (feature_where_no_zero.size() > torch.Size([0])):
                nodes_protein_name_dict[G_node] = node_label[feature_where_no_zero]
    for node in G.nodes():
        labels[node] = nodes_protein_name_dict[node]
    nx.draw_networkx_nodes(G, pos)   
    nx.draw_networkx_edges(G, pos)   
    nx.draw_networkx_labels(G, pos, labels)  
    nx.draw_networkx_labels(G, pos)
    plt.axis('off')              
    plt.show()
    plt.savefig('aaa.png')



def edgeindex_to_adj(edge_index, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for (start_idx, end_idx) in zip(edge_index[0].numpy(), edge_index[1].numpy()):
        adj_matrix[start_idx][end_idx] = 1
    return adj_matrix


def process_graph(adj, num_nodes):
    adj = edgeindex_to_adj(adj, num_nodes)
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.from_numpy(adj_normalized).float()


def Cartesian_2_Spherical(atom1, atom2_list):
    relative_position =list()
    if atom2_list.dim() == 1:
        x,y,z =atom2_list - atom1
        r = np.sqrt(x*x + y* y +z*z).numpy()
        theta = np.arccos(z/r).numpy()
        phi = np.arctan2(np.array(y), np.array(x))
        relative_position.append([r, theta, phi])
        return relative_position
    
    for atom in atom2_list:
        x,y,z =atom - atom1
        r = np.sqrt(x*x + y* y +z*z).numpy()
        theta = np.arccos(z/r).numpy()
        phi = np.arctan2(np.array(y), np.array(x))
        relative_position.append([r, theta, phi])
    return relative_position


def Spherical_2_Cartesian(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    arr= []
    arr.append(x)
    arr.append(y)
    if np.array([z]).shape == x.shape:
        arr.append(np.array([z]))
    else:
        arr.append(np.array(z))
    return np.array(arr)