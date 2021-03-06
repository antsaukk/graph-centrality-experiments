import networkx as nx
import numpy as np
import scipy.sparse.linalg
import matplotlib
import scipy.io
import mat73
import math

from matplotlib import pyplot as plt
from dataclasses import dataclass
from itertools import count


@dataclass
class CentralityAttributes:
    AdjacencyMatrix: scipy.sparse._csc.csc_matrix
    Graph: nx.classes.graph.Graph
    
def Reshape(vector: np.ndarray) -> np.ndarray:
    return np.reshape(vector, vector.shape[0])

def KatzGrid(spectral_rad: float,
             grid_points: int=9) -> np.ndarray:
    alphas = np.linspace(0, 1/spectral_rad, num=grid_points)
    alphas = alphas[1:len(alphas)-1]
    
    return alphas

def TGrid(minEigOfMt: float,
          grid_points: int=9) -> np.ndarray:
    ts = np.linspace(0, minEigOfMt, num=grid_points)
    #ts = np.linspace(0, spectral_rad, num=grid_points)
    ts = ts[1:len(ts)-1]
    #ts = ts[1:len(ts)]
    
    return ts

def SpectralRadius(A) -> float:
    W, V  = scipy.sparse.linalg.eigs(A)
    eigen = max(abs(W))
    lambd = float(eigen)

    return lambd

def SmallestSpectralRadius(A) -> float:
    W, V  = scipy.sparse.linalg.eigs(A)
    eigen = min(abs(W))
    lambd = float(eigen)

    return lambd

def DeformedGraphLaplacian(A,
                           I: np.ndarray,
                           D: np.ndarray,
                           t: float) -> np.matrix:
    
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
    N                   = A.shape[0]
    
    largest_eigen_value = SpectralRadius(A)
    
    alphas              = KatzGrid(largest_eigen_value)
    
    I                   = np.eye(N)
    e                   = np.ones((N, 1))
    
    katz_centralities   = np.zeros((N, len(alphas)))
    condition_numbers   = np.zeros(len(alphas))

    for i, alpha in enumerate(alphas):
        centr                   = np.linalg.inv(I - alpha*A) * e
        katz_centralities[:, i] = Reshape(centr)
        condition_numbers[i]    = np.linalg.cond(I - alpha*A)
    
    return (katz_centralities, alphas, condition_numbers)

def NBTCentrality(A) -> tuple((np.ndarray, np.ndarray)):
    N                          = A.shape[0]
    
    #largest_eigen_value        = SpectralRadius(A)
    #smallest_eigen_value        = SmallestSpectralRadius(A)
    
    ts                         = TGrid(0.0334) # 0.3214 for GD; 0.0334 for Myc
    
    nbtw_centralities          = np.zeros((N, len(ts)))
    condition_numbers          = np.zeros(len(ts))
    
    I                          = np.eye(N)
    e                          = np.ones((N, 1))
    
    d                          = np.sum(A, axis=1)
    D                          = np.eye(N)
    D[np.diag_indices_from(D)] = Reshape(d)

    for i, t in enumerate(ts):
        Mt                      = DeformedGraphLaplacian(A, I, D, t)
        
        centr_nbt               = np.linalg.inv(Mt) * e * (1 - t**2)
        nbtw_centralities[:, i] = Reshape(centr_nbt)
        condition_numbers[i]    = np.linalg.cond(Mt)
        
    return (nbtw_centralities, ts, condition_numbers)

def GenerateGroups(range_to_group: np.ndarray) -> tuple((list, np.ndarray)):
    lb           = min(range_to_group)
    ub           = max(range_to_group)
    
    intervals    = np.linspace(lb, ub, 6)
    
    boolmap      = list(map(lambda x: x > intervals, range_to_group))
    group_labels = list(map(lambda x: np.where((x > intervals) == False)[0][0],  range_to_group))
    
    # intr_labels  = ["Category boundaries:" + 
    #                 str(i) + 
    #                 " - {" + 
    #                 str((intervals[i-1])) + 
    #                 " " + 
    #                 str((intervals[i]))
    #                 + "}" 
    #                 for i in range(1, len(intervals))]
    # #print(intr_labels)
    
    return (group_labels, intervals)

def GenerateAttributes(groups_lbl: list) -> dict: 
    group_attr = {}
    for node in range(len(groups_lbl)):
        group_attr[node] = {'color_group': str(groups_lbl[node])}

    return group_attr

def CentralityColorMap(Graph: nx.classes.graph.Graph,
                       intervals: list,
                       param: float,
                       pos,
                       xsz: int,
                       ysz: int,
                       filename="") -> None:
    
    # get unique groups
    groups  = set(nx.get_node_attributes(Graph,'color_group').values())
    mapping = dict(zip(sorted(groups), count()))
    nodes   = Graph.nodes()
    colors  = [mapping[Graph.nodes[n]['color_group']] for n in nodes]

    # drawing nodes and edges separately so we can capture collection for colobar
    fig = plt.figure(figsize=(ysz,xsz))
    ec  = nx.draw_networkx_edges(Graph,
                                 pos,
                                 alpha=0.2)
    nc  = nx.draw_networkx_nodes(Graph,
                                 pos,
                                 nodelist=nodes,
                                 node_color=colors, 
                                 node_size=15,
                                 cmap=plt.cm.jet)
    lb  = nx.draw_networkx_labels(Graph,
                                  pos,
                                  font_size=9,
                                  font_color='k')
    
    cbar = plt.colorbar(nc, ticks=range(0,len(intervals)))
    cbar.ax.set_yticklabels(list(map(str, list(map(lambda x: round(x, 3), intervals)))))
    cbar.set_label('Colormap')
    plt.title(f"Centrality measure of the node as function of color with parameter {round(param, 4)}")
    plt.axis('off')
    if filename != "":
        plt.savefig(filename, format="jpg", bbox_inches="tight")
    plt.show()
    
def ShowConditionNumber(grid: np.ndarray,
                        condition_num: np.ndarray,
                        title: str,
                        filename: str) -> None:
    fig = plt.figure(figsize=(5,5))
    plt.plot(grid, condition_num)
    plt.title(f"Condition number of {title}")
    plt.xlabel('a')
    plt.ylabel(f"K({title})")
    plt.savefig(filename, format="jpg", bbox_inches="tight")

def VisualizeNodeCentrality(centrality_vector: np.ndarray,
                            grid: np.ndarray,
                            title: str,
                            xlab: str,
                            ylab: str,
                            xsz: int=5,
                            ysz: int=5,
                            step: int=1,
                            filename: str="") -> None:
    legend_list = []
    
    fig = plt.figure(figsize=(ysz, xsz))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    for i in range(0, centrality_vector.shape[0], step):
        lines = plt.plot(grid, centrality_vector[i, :], linewidth=0.5)
        legend_list.append('Node ' + str(i))
        
    matplotlib.rcParams['legend.fontsize'] = 5
    fig.legend(legend_list, loc=5)
    plt.grid()
    
    if filename != "":
        plt.savefig(filename, format="jpg", bbox_inches="tight")
        
    plt.show()
    
def DisplayCentralitiesInGraph(centrality_matrix: np.ndarray,
                               Graph: nx.classes.graph.Graph,
                               grid: np.ndarray,
                               xsz: int=15,
                               ysz: int=15,
                               filename="") -> None:
    
    fln = ""
    pos = nx.spring_layout(Graph)
    for i in range(centrality_matrix.shape[1]):
        centrality_vector = centrality_matrix[:, i]
        
        glbl, ints        = GenerateGroups(centrality_vector)
        test_attribute    = GenerateAttributes(glbl)
        
        nx.set_node_attributes(Graph, test_attribute)
        
        if filename != "":
            fln = filename + str(i) + ".jpg"
            
        CentralityColorMap(Graph, ints, grid[i], pos, xsz, ysz, fln)
        fln = ""