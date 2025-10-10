import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import random

def Event_masking(data):
    edge_index = data.edge_index
    NodeType = data.node_type
    X = data.x
    VariableX = data.variableX
    SerX = data.ServiceX
    EdgeType = data.edge_type
    EdgeAttr = data.edge_attr
    y = data.y

    new_X=[]
    new_VariableX=[]
    new_NodeType=[]

    PreIndex=[]
    TarIndex=[]
    new_SerX=[]
    new_EdgeType=[]
    new_EdgeAttr=[]

    mask_num = int(X.shape[0]*0.1)
    x_idx_mask = np.random.choice(X.shape[0], mask_num, replace=False)
    variablex_idx_mask = np.random.choice(X.shape[0], mask_num, replace=False)


    for i in range(X.shape[0]):
        if i in x_idx_mask:
            newX=np.random.randn(1,X.shape[1]).squeeze()
            new_X.append(newX)
        else:
            new_X.append(X[i].numpy())

        if i in variablex_idx_mask:
            newVariableX=np.random.randn(1,VariableX.shape[1]).squeeze()
            new_VariableX.append(newVariableX)
        else:
            new_VariableX.append(VariableX[i].numpy())

        new_NodeType.append(NodeType[i].item())
        new_SerX.append(SerX[i].numpy())

    for i in range(edge_index.shape[1]):
        PreIndex.append(edge_index[0][i].item())
        TarIndex.append(edge_index[1][i].item())
        new_EdgeType.append(EdgeType[i].item())
        new_EdgeAttr.append(EdgeAttr[i].numpy())

    edge_index = torch.as_tensor([PreIndex, TarIndex], dtype=torch.long)
    X = torch.as_tensor(np.array(new_X), dtype=torch.float64)
    VariableX = torch.as_tensor(np.array(new_VariableX), dtype=torch.float64)
    SerX = torch.as_tensor(np.array(new_SerX), dtype=torch.float64)
    NodeType = torch.as_tensor(np.array(new_NodeType), dtype=torch.long)
    EdgeType = torch.as_tensor(np.array(new_EdgeType), dtype=torch.long)
    EdgeAttr = torch.as_tensor(np.array(new_EdgeAttr), dtype=torch.float64)

    data_aug = Data(x=X, variableX=VariableX, ServiceX=SerX, node_type=NodeType, edge_type=EdgeType,
                    edge_index=edge_index, edge_attr=EdgeAttr, y=y)

    return data_aug

def Event_Metric_masking(data):
    edge_index = data.edge_index
    NodeType = data.node_type
    X = data.x
    VariableX = data.variableX
    SerX = data.ServiceX
    EdgeType = data.edge_type
    EdgeAttr = data.edge_attr
    y = data.y

    new_X = []
    new_VariableX = []
    new_NodeType = []

    PreIndex = []
    TarIndex = []
    new_SerX = []
    new_EdgeType = []
    new_EdgeAttr = []

    mask_num = int(X.shape[0]*0.2)
    idx_mask = np.random.choice(X.shape[0], mask_num, replace=False)

    for i in range(X.shape[0]):
        if i in idx_mask:
            new_X.append(X[i].numpy())
            new_VariableX.append(VariableX[i].numpy())
            new_NodeType.append(NodeType[i].item())
            newSerX=np.random.randn(1,SerX.shape[1]).squeeze()
            new_SerX.append(newSerX)
        else:
            new_X.append(X[i].numpy())
            new_VariableX.append(VariableX[i].numpy())
            new_NodeType.append(NodeType[i].item())
            new_SerX.append(SerX[i].numpy())

    for i in range(edge_index.shape[1]):
        PreIndex.append(edge_index[0][i].item())
        TarIndex.append(edge_index[1][i].item())
        new_EdgeType.append(EdgeType[i].item())
        new_EdgeAttr.append(EdgeAttr[i].numpy())

    edge_index = torch.as_tensor([PreIndex, TarIndex], dtype=torch.long)
    X = torch.as_tensor(np.array(new_X), dtype=torch.float64)
    VariableX = torch.as_tensor(np.array(new_VariableX), dtype=torch.float64)
    SerX = torch.as_tensor(np.array(new_SerX), dtype=torch.float64)
    NodeType = torch.as_tensor(np.array(new_NodeType), dtype=torch.long)
    EdgeType = torch.as_tensor(np.array(new_EdgeType), dtype=torch.long)
    EdgeAttr = torch.as_tensor(np.array(new_EdgeAttr), dtype=torch.float64)

    data_aug = Data(x=X, variableX=VariableX, ServiceX=SerX, node_type=NodeType, edge_type=EdgeType,
                    edge_index=edge_index, edge_attr=EdgeAttr, y=y)

    return data_aug

def invocation_Metric_masking(data):
    edge_index = data.edge_index
    NodeType = data.node_type
    X = data.x
    VariableX = data.variableX
    SerX = data.ServiceX
    EdgeType = data.edge_type
    EdgeAttr = data.edge_attr
    y = data.y

    new_X = []
    new_VariableX = []
    new_NodeType = []

    PreIndex = []
    TarIndex = []
    new_SerX = []
    new_EdgeType = []
    new_EdgeAttr = []

    for i in range(X.shape[0]):
        new_X.append(X[i].numpy())
        new_VariableX.append(VariableX[i].numpy())
        new_NodeType.append(NodeType[i].item())
        new_SerX.append(SerX[i].numpy())

    mask_num = int(edge_index.shape[1]*0.2)
    idx_mask = np.random.choice(edge_index.shape[1], mask_num, replace=False)

    for i in range(edge_index.shape[1]):
        PreIndex.append(edge_index[0][i].item())
        TarIndex.append(edge_index[1][i].item())
        new_EdgeType.append(EdgeType[i].item())

        if i in idx_mask:
            newEdgeAttr=np.random.randn(1,EdgeAttr.shape[1]).squeeze()
            new_EdgeAttr.append(newEdgeAttr)
        else:
            new_EdgeAttr.append(EdgeAttr[i].numpy())

    edge_index = torch.as_tensor([PreIndex, TarIndex], dtype=torch.long)
    X = torch.as_tensor(np.array(new_X), dtype=torch.float64)
    VariableX = torch.as_tensor(np.array(new_VariableX), dtype=torch.float64)
    SerX = torch.as_tensor(np.array(new_SerX), dtype=torch.float64)
    NodeType = torch.as_tensor(np.array(new_NodeType), dtype=torch.long)
    EdgeType = torch.as_tensor(np.array(new_EdgeType), dtype=torch.long)
    EdgeAttr = torch.as_tensor(np.array(new_EdgeAttr), dtype=torch.float64)

    data_aug = Data(x=X, variableX=VariableX,ServiceX=SerX, node_type=NodeType, edge_type=EdgeType,
                    edge_index=edge_index, edge_attr=EdgeAttr, y=y)

    return data_aug

def invocation_swap_nodes(data):
    edge_index = data.edge_index
    NodeType = data.node_type
    X = data.x
    VariableX = data.variableX
    SerX = data.ServiceX
    EdgeType = data.edge_type
    EdgeAttr = data.edge_attr
    y = data.y

    new_X = []
    new_VariableX = []
    new_NodeType = []

    PreIndex = []
    TarIndex = []
    new_SerX = []
    new_EdgeType = []
    new_EdgeAttr = []

    for i in range(X.shape[0]):
        new_X.append(X[i].numpy())
        new_VariableX.append(VariableX[i].numpy())
        new_NodeType.append(NodeType[i].item())
        new_SerX.append(SerX[i].numpy())

    mask_num = random.randint(0,int(edge_index.shape[1]*0.2))
    idx_mask = np.random.choice(edge_index.shape[1], mask_num, replace=False)

    for i in range(edge_index.shape[1]):
        if i in idx_mask:
            PreIndex.append(edge_index[1][i].item())
            TarIndex.append(edge_index[0][i].item())
        else:
            PreIndex.append(edge_index[0][i].item())
            TarIndex.append(edge_index[1][i].item())

        new_EdgeType.append(EdgeType[i].item())
        new_EdgeAttr.append(EdgeAttr[i].numpy())

    edge_index = torch.as_tensor([PreIndex, TarIndex], dtype=torch.long)
    X = torch.as_tensor(np.array(new_X), dtype=torch.float64)
    VariableX = torch.as_tensor(np.array(new_VariableX), dtype=torch.float64)
    SerX = torch.as_tensor(np.array(new_SerX), dtype=torch.float64)
    NodeType = torch.as_tensor(np.array(new_NodeType), dtype=torch.long)
    EdgeType = torch.as_tensor(np.array(new_EdgeType), dtype=torch.long)
    EdgeAttr = torch.as_tensor(np.array(new_EdgeAttr), dtype=torch.float64)

    data_aug = Data(x=X, variableX=VariableX, ServiceX=SerX, node_type=NodeType, edge_type=EdgeType,
                    edge_index=edge_index, edge_attr=EdgeAttr, y=y)

    return data_aug


def invocation_interruption(data):
    edge_index = data.edge_index
    NodeType = data.node_type
    X = data.x
    VariableX = data.variableX
    SerX = data.ServiceX
    EdgeType = data.edge_type
    EdgeAttr = data.edge_attr
    y = data.y

    new_X = []
    new_VariableX = []
    new_NodeType = []

    PreIndex = []
    TarIndex = []
    new_SerX = []
    new_EdgeType = []
    new_EdgeAttr = []

    idx = random.randint(0,X.shape[0]-1)
    rev_nodes, sub_edge_index, drop_nodes, mask_edge = k_hop_subgraph(node_idx=[idx], num_hops=int(edge_index.shape[1]*0.2+1),
                                                                      edge_index=data.edge_index)
    rev_nodes=list(rev_nodes.numpy())

    for i in range(X.shape[0]):
        if i in rev_nodes:
            new_X.append(X[i].numpy())
            new_VariableX.append(VariableX[i].numpy())
            new_NodeType.append(NodeType[i].item())
            new_SerX.append(SerX[i].numpy())

    for i in range(edge_index.shape[1]):
        if edge_index[0][i].item() in rev_nodes and edge_index[1][i].item() in rev_nodes:
            PreIndex.append(edge_index[0][i].item())
            TarIndex.append(edge_index[1][i].item())
            new_EdgeType.append(EdgeType[i].item())
            new_EdgeAttr.append(EdgeAttr[i].numpy())

    nodes_idx_dict = {rev_nodes[i]: i for i in list(range(len(rev_nodes)))}
    newPreIndex=[ nodes_idx_dict[n] for n in PreIndex]
    newTarIndex=[nodes_idx_dict[n] for n in TarIndex]

    edge_index = torch.as_tensor([newPreIndex, newTarIndex], dtype=torch.long)
    X = torch.as_tensor(np.array(new_X), dtype=torch.float64)
    VariableX = torch.as_tensor(np.array(new_VariableX), dtype=torch.float64)
    SerX = torch.as_tensor(np.array(new_SerX), dtype=torch.float64)
    NodeType = torch.as_tensor(np.array(new_NodeType), dtype=torch.long)
    EdgeType = torch.as_tensor(np.array(new_EdgeType), dtype=torch.long)
    EdgeAttr = torch.as_tensor(np.array(new_EdgeAttr), dtype=torch.float64)

    data_aug = Data(x=X, variableX=VariableX, ServiceX=SerX, node_type=NodeType, edge_type=EdgeType,
                    edge_index=edge_index, edge_attr=EdgeAttr, y=y)

    return data_aug