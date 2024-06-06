import torch
import torch_geometric
import torch.nn as nn 
from torch_geometric.data import Data, Batch
from torch_geometric.utils import grid
from torch_geometric.nn import Sequential
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def _get_neighbors(x, y, height, width):
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < height - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < width - 1:
        neighbors.append((x, y + 1))
    return neighbors


def binary_map_to_graph(binary_map):
    if isinstance(binary_map, np.ndarray):
        binary_map = torch.tensor(binary_map, dtype=torch.float32)
    elif not isinstance(binary_map, torch.Tensor):
        raise ValueError("Input must be a numpy array or a torch tensor.")

    if binary_map.dim() != 2:
        raise ValueError("Binary map must be a 2D array or tensor.")

    h, w = binary_map.shape
    x_index_dict = dict()

    x = []
    edge_index = [[],[]]

    for i in range(h):
        for j in range (w):
            if binary_map[i,j] == 1:
                x_index = len(x)
                x_index_dict[(i,j)] = x_index
                x.append((i,j))

    for i in range(h):
        for j in range(w):
            if binary_map[i,j] == 1:

                x_index = x_index_dict[(i,j)]
                neighbors = _get_neighbors(i,j,h,w)
                
                for neighbor in neighbors:
                    if binary_map[neighbor] == 1:
                        neighbor_index = x_index_dict[neighbor]
                        edge_index[0].append(x_index)
                        edge_index[1].append(neighbor_index)
    
    # Create PyTorch Geometric data object
    data = Data(x=torch.Tensor(x), edge_index=torch.LongTensor(edge_index))
    
    return data

def visualize_map(tensor, title="Binary Tensor Visualization"):
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")
    
    # Convert tensor to numpy array
    tensor_np = tensor.numpy()

    # Plot the binary tensor
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_graph(data):
    # Convert PyTorch Geometric Data to NetworkX graph
    G = nx.Graph()

    pos = data.x.numpy()
    edge_index = data.edge_index.numpy()

    for i, (x, y) in enumerate(pos):
        G.add_node(i, pos=(y, -x))  # Use (y, -x) to match the image coordinates

    for i, j in edge_index.T:
        G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_weight='bold')
    plt.show()

class MIMRGNN(nn.Module):
    def __init__(
            self,
            layer_num:int,
            inter_dim:int,
            out_dim:int,
        )->torch.Tensor :
        super().__init__()
        conv_modules_list = []

        self.layer_num= layer_num
        self.map_size_limit = 100000

        conv_modules_list.append((torch_geometric.nn.GCNConv(2,inter_dim),'x, edge_index->x'))
        for i in range(layer_num):
            conv_modules_list.append(
                (torch_geometric.nn.GCNConv(inter_dim, inter_dim),'x, edge_index->x'),
            )
            conv_modules_list.append(nn.ReLU())
        conv_modules_list.append((torch_geometric.nn.GCNConv(inter_dim,out_dim),'x, edge_index->x'))

        self.conv_modules = Sequential('x, edge_index', conv_modules_list)
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

    def _preprocess(self, x):
        # input: b, h, w, c(binary map)
        # output: graph batch

        b, h, w, c = x.shape

        map_graphs = []
        for i in range(b):
            binary_map = x[i,:,:]
            _, h, w = binary_map.shape

            #visualize_map(binary_map.squeeze().detach().cpu())
            while h*w > self.map_size_limit:
                binary_map = self.maxpool2d(binary_map)
                binary_map = binary_map.round()
                _, h, w = binary_map.shape
            
            map_graph = binary_map_to_graph(binary_map.squeeze())
            map_graphs.append(map_graph)
            map_graphs_batch = Batch.from_data_list(map_graphs)

        return map_graphs_batch


    def forward(self, map):
        
        batch = self._preprocess(map)

        x = batch.x.cuda()
        edge_index = batch.edge_index.cuda()
        b = batch.batch.cuda()
        
        x = self.conv_modules(x, edge_index=edge_index)
        
        return torch.sigmoid(torch_geometric.nn.global_mean_pool(x, b))

            
            






# Example usage
if __name__ == "__main__":
    # Dummy binary map
    binary_map = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0]
    ], dtype=np.float32)

    graph_data = binary_map_to_graph(binary_map)
    print(graph_data)
    print("Number of nodes:", graph_data.num_nodes)
    print("Number of edges:", graph_data.num_edges)
    print("Edge index:", graph_data.edge_index)

    # Visualize the graph
    visualize_graph(graph_data)
