import networkx as nx
import numpy as np
import scipy.sparse.linalg
import scipy.io
import mat73
import math
import h5py

from matplotlib import pyplot as plt
from dataclasses import dataclass
from itertools import count

@dataclass
class CentralityAttributes:
    AdjacencyMatrix: scipy.sparse._csc.csc_matrix
    Graph: nx.classes.graph.Graph
    
def SpectralRadius(A) -> float:
    W, V   = scipy.sparse.linalg.eigs(A) # get largest eigenvalue of adj matrix
    eigens = sorted(W, reverse=True)
    lambd  = float(eigens[0])
    
    return lambd

def KatzGrid(spectral_rad: float, grid_points: int=9) -> np.ndarray:
    alphas = np.linspace(0, 1/spectral_rad, num=grid_points)
    alphas = alphas[1:len(alphas)-1]
    
    return alphas

def TGrid(grid_points: int=9) -> np.ndarray:
    ts = np.linspace(0, 1, num=grid_points)
    ts = ts[1:len(ts)-1]
    
    return ts

def DeformedGraphLaplacian(A, I: np.ndarray, D: np.ndarray, t: float) -> np.matrix:
    return I - t*A + (D - I) * t**2

def KatzCentralityNX(A, Graph: nx.classes.graph.Graph) -> tuple((np.ndarray, np.ndarray)):
    largest_eigen_value = SpectralRadius(A)
    
    alphas = KatzGrid(largest_eigen_value)
    
    katz_centralities = np.zeros((A.shape[0], len(alphas)))

    for i, alpha in enumerate(alphas):
        cent                    = nx.katz_centrality(Graph, alpha=alpha)
        katz_centralities[:, i] = np.array(list(cent.values())).astype(float)
        
    return (katz_centralities, alphas)

def KatzCentralityV2(A, Graph: nx.classes.graph.Graph) -> tuple((np.ndarray, np.ndarray)):
    largest_eigen_value = SpectralRadius(A)
    
    alphas = KatzGrid(largest_eigen_value)
    
    I      = np.eye(A.shape[0])
    e      = np.ones((A.shape[0], 1))
    
    katz_centralities = np.zeros((A.shape[0], len(alphas)))

    for i, alpha in enumerate(alphas):
        centr                   = np.linalg.inv(I - alpha*A) * e
        katz_centralities[:, i] = np.reshape(centr, centr.shape[0])
    
    return (katz_centralities, alphas)

def NBTCentrality(A) -> tuple((np.ndarray, np.ndarray)):
    ts                         = TGrid()
    
    nbtw_centralities          = np.zeros((A.shape[0], len(ts)))
    
    I                          = np.eye(A.shape[0], A.shape[0])
    e                          = np.ones((A.shape[0], 1))
    
    d                          = np.sum(A, axis=1)
    D                          = np.identity(A.shape[0])
    D[np.diag_indices_from(D)] = np.reshape(d, d.shape[0])

    for i, t in enumerate(ts):
        Mt                      = DeformedGraphLaplacian(A, I, D, t)
        
        centr_nbt               = np.linalg.inv(Mt) * e * (1 - t**2)
        nbtw_centralities[:, i] = np.reshape(centr_nbt, centr_nbt.shape[0])
        
    return (nbtw_centralities, ts)

def GenerateGroups(range_to_group: np.ndarray) -> tuple((list, np.ndarray)):
    lb           = min(range_to_group)
    ub           = max(range_to_group)
    
    intervals    = np.linspace(lb, ub, 6)
    
    boolmap      = list(map(lambda x: x > intervals, range_to_group))
    group_labels = list(map(lambda x: np.where((x > intervals) == False)[0][0],  range_to_group))
    
    return (group_labels, intervals)

def GenerateAttributes(groups_lbl: list) -> dict: 
    group_attr = {}
    for node in range(len(groups_lbl)):
        group_attr[node] = {"color_group": str(groups_lbl[node])}

    return group_attr

def CentralityColorMap(Graph: nx.classes.graph.Graph) -> None:
    # get unique groups
    groups  = set(nx.get_node_attributes(Graph,'color_group').values())
    mapping = dict(zip(sorted(groups), count()))
    nodes   = Graph.nodes()
    colors  = [mapping[Graph.nodes[n]['color_group']] for n in nodes]

    # drawing nodes and edges separately so we can capture collection for colobar
    fig = plt.figure(figsize=(15,15))
    pos = nx.spring_layout(Graph)
    ec  = nx.draw_networkx_edges(Graph, pos, alpha=0.2)
    nc  = nx.draw_networkx_nodes(Graph, pos, nodelist=nodes, node_color=colors, 
                                 node_size=15, cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()

def VisualizeNodeCentrality(centrality_vector: np.ndarray, grid: np.ndarray) -> None:
    fig1 = plt.figure(figsize=(10, 10))
    for i in range(centrality_vector.shape[0]):
        plt.plot(grid, centrality_vector[i, :])
    plt.show()
    
def DisplayCentralitiesInGraph(centrality_matrix: np.ndarray, Graph: nx.classes.graph.Graph) -> None:
    
    for i in range(centrality_matrix.shape[1]):
        centrality_vector = centrality_matrix[:, i]
        
        glbl, ints        = GenerateGroups(centrality_vector)
        test_attribute    = GenerateAttributes(glbl)
        
        nx.set_node_attributes(Graph, test_attribute)
        CentralityColorMap(Graph)
        print(ints)