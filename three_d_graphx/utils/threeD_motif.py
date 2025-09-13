import networkx as nx
import numpy as np

#find all the motifs
def tree_decompose(atom_names, all_edges):
    G = nx.Graph()
    G.add_nodes_from(list(atom_names))
    G.add_edges_from(all_edges)
    #draw(G,feat)

    #find all the rings
    total_ring=nx.cycle_basis(G)
    total_ring = sorted(total_ring, key=len, reverse=False)

    #Store all the edges that are not in the rings
    clusters = []

    for (u, v) in G.edges:
        for each_ring in total_ring:
            if u in each_ring and v in each_ring:
                break
        else:
            clusters.append({u,v})
    
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            commoned_atom = clusters[i] & clusters[j]
            if len(commoned_atom)==0 : continue
            atom1, atom2 = list(clusters[i] -commoned_atom)[0],list(clusters[j] -commoned_atom)[0]
            if isinstance(atom1,np.int64) and isinstance(atom2, np.int64):
                if G.degree(atom1) == 1 and G.degree(atom2) == 1:
                    clusters[i] = clusters[i] | clusters[j]
                    clusters[j] = set()
            if isinstance(atom1, list) or isinstance(atom2, list):
                flag = True
                if isinstance(atom1, list):
                    for atom_in in atom1:
                        if G.degree(atom_in) != 1:
                            flag = False
                if isinstance(atom1, list):
                    for atom_in in atom2:
                        if G.degree(atom_in) != 1:
                            flag = False
                if flag:
                    clusters[i] = clusters[i] | clusters[j]
                    clusters[j] = set()
                    
    clusters =[list(c) for c in clusters if len(c) > 0]

    for each_ring in total_ring:
        clusters.append(each_ring)

    rotable = dict()
    for i, c in enumerate(clusters):
        rotable[i] = False
        clique = [x for x in c]
        if len(clique) == 2:
            if G.degree(clique[0]) >= 2 and G.degree(clique[1]) >= 2:
                rotable[i] = True
    return clusters, rotable