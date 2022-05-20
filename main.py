import networkx as nx
import centrality_utils as cu
import mat73

from matplotlib import pyplot as plt


def main():
    Data    = mat73.loadmat('mycielskian8.mat')
    A       = Data['Problem']['A']
    G0      = nx.Graph(A)
    CA      = cu.CentralityAttributes(A, G0)
    
    fig0 = plt.figure(figsize=(10,10))
    nx.draw(CA.Graph, with_labels=True, node_size=10, width=0.1)
    
    KatzNX, alpha_grid   = cu.KatzCentralityNX(CA.AdjacencyMatrix, CA.Graph)
    KatzV2, alpha_gridV2 = cu.KatzCentralityV2(CA.AdjacencyMatrix, CA.Graph)
    NBTcent, grid_t      = cu.NBTCentrality(CA.AdjacencyMatrix)
    
    cu.VisualizeNodeCentrality(KatzNX, alpha_grid)
    cu.VisualizeNodeCentrality(KatzV2, alpha_gridV2)
    cu.VisualizeNodeCentrality(NBTcent, grid_t)
    
    cu.DisplayCentralitiesInGraph(KatzV2, CA.Graph)
    cu.DisplayCentralitiesInGraph(NBTcent, CA.Graph)

if __name__ == "__main__":
    main()