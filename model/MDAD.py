import torch.nn as nn
from Microservices.AnomalyDetection.DALAD.model.MDADEmbedding import MDADEmbedding
from Microservices.AnomalyDetection.DALAD.model.VAE import VAE
import torch.nn.init as init

class MDAD(nn.Module):
    def __init__(self,num_layers,node_dim,variableX_dim,serviceX_dim,edge_dim,out_channels,num_edge_types, edge_type_emb_dim, edge_attr_emb_dim):
        super(MDAD, self).__init__()
        self.embedding = MDADEmbedding(num_layers,node_dim,variableX_dim,serviceX_dim,
                                       edge_dim,out_channels,num_edge_types,edge_type_emb_dim,edge_attr_emb_dim)
        self.vae1 = VAE(out_channels)
        self.vae2 = VAE(out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, oridata,data_neg):
        ori_hg=self.embedding(oridata)
        neg_hg=self.embedding(data_neg)
        ori_z,ori_hg_hat,ori_mu,ori_logvar= self.vae1(ori_hg)
        neg_z, neg_hg_hat, neg_mu, neg_logvar=self.vae2(neg_hg)
        return ori_z,neg_z,ori_hg,ori_hg_hat,ori_mu,ori_logvar,neg_hg,neg_hg_hat,neg_mu, neg_logvar
