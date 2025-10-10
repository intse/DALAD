import torch
import torch.nn as nn
from torch_geometric.nn import AttentionalAggregation,BatchNorm
from Microservices.AnomalyDetection.DALAD.model.MDADConv import MDADConv
import copy

class MDADEmbedding(nn.Module):
    def __init__(self,num_layers,node_dim,variableX_dim,serviceX_dim,edge_dim,out_channels,num_edge_types,edge_type_emb_dim,edge_attr_emb_dim):
        super(MDADEmbedding, self).__init__()
        self.out_channels_head=int(out_channels/4)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = MDADConv(in_channels=out_channels,out_channels=self.out_channels_head,num_node_types=2,
                            num_edge_types=num_edge_types,edge_type_emb_dim=edge_type_emb_dim,edge_dim=edge_dim,
                            edge_attr_emb_dim=edge_attr_emb_dim,heads=4)
            self.convs.append(conv)
        self.node_fc=nn.Sequential(
            nn.Linear(node_dim+variableX_dim+serviceX_dim,out_channels),
            nn.Tanh(),
            BatchNorm(out_channels)
        )
        self.node_activate = nn.Tanh()
        self.nn =nn.Sequential(
            nn.Linear(2*out_channels, out_channels),
            nn.Tanh(),
            BatchNorm(out_channels)
        )
        self.gate=nn.Sequential(
            nn.Linear(2 * out_channels, 1),
            nn.Tanh()
        )
        self.pool=AttentionalAggregation(gate_nn = self.gate,nn=self.nn)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x,variableX,serviceX,node_type, edge_index,edge_attr,edge_type=\
            data.x,data.variableX,data.ServiceX, data.node_type,data.edge_index,data.edge_attr,data.edge_type
        X = torch.cat((x,variableX, serviceX), dim=1)
        X = self.node_fc(X)
        X_copy = copy.copy(X)
        for conv in self.convs:
            aggr_message = conv(X,edge_index,node_type,edge_type,edge_attr)
            X=X+self.node_activate(aggr_message)
        Batch = data.batch
        outputs = torch.cat((X_copy, X), dim=1)
        hg = self.pool(outputs, Batch)
        return hg

