import torch
import numpy as np
import random
from Microservices.AnomalyDetection.DALAD.model.Trianer import MDADTrainer
from Microservices.AnomalyDetection.DALAD.model.MDAD import MDAD

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    for i in range(5):
        num_edge_types = 5
        edge_type_emb_dim = 5
        node_variable_dim = 300
        service_metric_dim = 9
        edge_dim = 17
        edge_attr_emb_dim = 300
        node_template_dim = 300
        batch_size = 32
        n_epochs = 10
        out_channels = 32
        n_layer = 1
        beta = 1.0
        k = 32
        lr = 0.0001
        wd = 0.0001
        model_path = "./Experiments/SN/"

        setup_seed(42+i)
        MDADTrainerI = MDADTrainer(batch_size, lr, n_epochs, model_path, beta, k, out_channels, wd, "SN")
        net = MDAD(n_layer, node_template_dim, node_variable_dim, service_metric_dim,
                   edge_dim, out_channels, num_edge_types, edge_type_emb_dim, edge_attr_emb_dim)
        net = net.double()
        net = MDADTrainerI.train(net)
        net.load_state_dict(torch.load(model_path+"/best_network.pth"))
        net = net.double()
        normal_gmm, anomaly_gmm = MDADTrainerI.train_GMM(net)
        MDADTrainerI.test(net,normal_gmm,anomaly_gmm)
