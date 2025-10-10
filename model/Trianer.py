import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from Microservices.AnomalyDetection.DALAD.dataset.TTDataset import TTDataset
from Microservices.AnomalyDetection.DALAD.dataset.SNDataset import SNDataset
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
import random
from Microservices.AnomalyDetection.DALAD.utils.early_stoping import EarlyStopping
from Microservices.observability.utils.log import Logger
import logging
from sklearn.mixture import GaussianMixture
import numpy as np
from Microservices.AnomalyDetection.DALAD.utils.GPUMemoryMonitor import GPUMemoryMonitor

class MDADTrainer():
    def __init__(self,batch_size,lr,n_epochs,model_path,beta,k,out_channels,weight_decay,dataName):
        super().__init__()
        self.device="cuda:0"
        self.dataName=dataName
        self.model_path=model_path
        self.log_path = self.model_path+"trainlog"
        self.logger = Logger(self.log_path, logging.INFO, __name__).getlog()

        if self.dataName=='TT':
            self.dataset=TTDataset("./OBD/TT/")
            self.train_index, self.val_index, self.test_index = self.TT_data_index_generate()
        else:
            self.dataset = SNDataset("./OBD/SN/")
            self.train_index, self.val_index, self.test_index = self.Other_data_index_generate(self.dataName)

        self.batch_size=batch_size
        self.lr=lr
        self.n_epochs=n_epochs
        self.beta = beta
        self.weight_decay = weight_decay
        self.k = k
        self.dim = out_channels
        self.train_time = None
        self.test_time = None
        self.normal_gmm = None
        self.anomaly_gmm = None

    def TT_data_index_generate(self):
        all_index = range(0, 167049)
        normal_index = range(15138, 167049)
        train_index = random.sample(normal_index, round(0.8 * len(normal_index)))
        normal_leave_index = set(normal_index) - set(train_index)
        val_index = random.sample(list(normal_leave_index), round(0.5 * len(normal_leave_index)))
        test_index = list(set(all_index) - set(train_index) - set(val_index))
        self.logger.info("Train dataset: " + str(len(train_index)))
        self.logger.info("Val dataset: " + str(len(val_index)))
        self.logger.info("Test dataset: " + str(len(test_index)))
        return train_index, val_index, test_index

    def Other_data_index_generate(self,dataName):
        normal_index = []
        abnormal_index = []
        with open("./OBD/"+dataName+"/normal_index.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n", "")
                normal_index.append(int(line))
        f.close()
        with open("./OBD/"+dataName+"/abnormal_index.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("\n", "")
                abnormal_index.append(int(line))
        f.close()
        train_index = random.sample(normal_index, round(0.8 * len(normal_index)))
        normal_leave_index = set(normal_index) - set(train_index)
        val_index = random.sample(list(normal_leave_index), round(0.5 * len(normal_leave_index)))
        test_index = list(set(normal_index) - set(train_index) - set(val_index)) + abnormal_index
        self.logger.info("Train dataset: " + str(len(train_index)))
        self.logger.info("Val dataset: " + str(len(val_index)))
        self.logger.info("Test dataset: " + str(len(test_index)))
        return train_index, val_index, test_index

    def train(self, net):
        net = net.to(self.device)
        GNN_total_params = sum(p.numel() for p in net.parameters())
        GMM_total_params = self.k*(self.dim*self.dim+3*self.dim+2)
        total_params = GNN_total_params + GMM_total_params
        self.logger.info(f'Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)')
        self.logger.info('Initializing train dataset and eval dataset...')
        train_dataset= self.dataset[self.train_index]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        val_dataset = self.dataset[self.val_index]
        val_loader=DataLoader(val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)
        self.logger.info('Train dataset and eval dataset initialized.')
        optimizer = optim.Adam(net.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        self.logger.info('Starting training...')
        start_time = time.time()
        train_loss_value = []
        cv_loss_value = []
        early_stopping = EarlyStopping(save_path=self.model_path,logger=self.logger,patience=5,delta=0.001)
        train_monitor = GPUMemoryMonitor()
        train_monitor.start()
        for epoch in range(self.n_epochs):
            loss_epoch = 0.0
            n_step=0
            net.train()
            epoch_start_time = time.time()
            for data in train_loader:
                oridata,data_rc = data
                oridata=oridata.to(self.device)
                data_rc = data_rc.to(self.device)
                optimizer.zero_grad()

                hg_hidden, neg_hg_hidden, ori_hg, ori_hg_hat, ori_mu, \
                ori_logvar, neg_hg, neg_hg_hat, neg_mu, neg_logvar = net(oridata,data_rc)
                loss = self.vae_loss_cal(ori_hg, ori_hg_hat, ori_mu, ori_logvar, neg_hg, neg_hg_hat, neg_mu,neg_logvar)

                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                n_step+=1
            epoch_train_time = time.time() - epoch_start_time
            self.logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.10f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_step))
            train_loss_value.append(loss_epoch / n_step)
            net.eval()
            cv_loss = 0
            n_step=0
            with torch.no_grad():
                for data in val_loader:
                    oridata, data_rc = data
                    oridata = oridata.to(self.device)
                    data_rc = data_rc.to(self.device)

                    hg_hidden, neg_hg_hidden, ori_hg, ori_hg_hat, ori_mu, \
                    ori_logvar, neg_hg, neg_hg_hat, neg_mu, neg_logvar = net(oridata, data_rc)
                    loss = self.vae_loss_cal(ori_hg, ori_hg_hat, ori_mu, ori_logvar, neg_hg, neg_hg_hat, neg_mu,neg_logvar)
                    cv_loss += loss.item()
                    n_step+=1
            self.logger.info('Epoch: %d,  CV Loss: %.10f' % (epoch + 1, cv_loss / n_step))
            cv_loss_value.append(cv_loss / n_step)
            early_stopping(cv_loss / n_step, net)
            if early_stopping.early_stop:
                self.logger.info("Early stopping!!")
                break

        peak_train_memory = train_monitor.stop()
        self.logger.info(f'Peak GPU memory during training: {peak_train_memory:.2f} MB')
        self.train_time = time.time() - start_time
        self.logger.info('GNN Training time: %.3f' % self.train_time)
        self.logger.info('Finished training.')

        x = [i for i in range(0, epoch+1)]
        plt.title('Loss vs. epoches')
        plt.ylabel('Loss')
        plt.xlabel("Epoches")
        plt.plot(x, train_loss_value, marker='o', color='y', markersize=5)
        plt.plot(x, cv_loss_value, marker='x', color='blue', markersize=5)
        plt.legend(("Train_Loss", "CV_Loss"))
        plt.savefig(self.model_path+'TrainVsCv.png', dpi=120)
        plt.show()
        return net

    def train_GMM(self,net):
        net = net.to(self.device)
        train_dataset= self.dataset[self.train_index]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        train_normal_embeddings = []
        train_anomaly_embeddings = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                oridata, data_rc= data
                oridata = oridata.to(self.device)
                data_rc = data_rc.to(self.device)
                hg_hidden, neg_hg_hidden, ori_hg, ori_hg_hat, ori_mu, \
                ori_logvar, neg_hg, neg_hg_hat, neg_mu, neg_logvar = net(oridata, data_rc)
                train_normal_embeddings.append(hg_hidden)
                train_anomaly_embeddings.append(neg_hg_hidden)

        train_normal_embeddings = torch.cat(train_normal_embeddings, dim=0)
        train_anomaly_embeddings = torch.cat(train_anomaly_embeddings, dim=0)
        self.normal_gmm, self.anomaly_gmm = self.GaussianMixture_model(train_normal_embeddings.cpu().detach().numpy(),
                                                                       train_anomaly_embeddings.cpu().detach().numpy())
        self.GMM_time = time.time() - start_time
        self.logger.info('GMM Training time: %.3f' % self.GMM_time)
        return self.normal_gmm, self.anomaly_gmm


    def test(self, net,normal_gmm,anomaly_gmm):
        net = net.to(self.device)
        self.normal_gmm, self.anomaly_gmm = normal_gmm,anomaly_gmm
        test_dataset = self.dataset[self.test_index]
        test_loader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=8)
        self.logger.info('Starting testing...')
        test_embeddings = []
        test_labels = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        time.sleep(1.0)
        test_monitor = GPUMemoryMonitor()
        test_monitor.start()

        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                oridata,data_rc = data
                oridata = oridata.to(self.device)
                hg_hidden, neg_hg_hidden, ori_hg, ori_hg_hat, ori_mu, \
                ori_logvar, neg_hg, neg_hg_hat, neg_mu, neg_logvar = net(oridata, oridata)
                test_embeddings.append(hg_hidden)
                test_labels.append(oridata.y)

        peak_test_memory = test_monitor.stop()
        self.logger.info(f'Peak GPU memory during testing: {peak_test_memory:.2f} MB')
        test_embeddings = torch.cat(test_embeddings, dim=0).cpu().detach().numpy()
        normal_log_likelihoods = self.normal_gmm.score_samples(test_embeddings)
        anomaly_log_likelihoods = self.anomaly_gmm.score_samples(test_embeddings)
        preds = np.where(normal_log_likelihoods > anomaly_log_likelihoods, 0, 1)
        self.test_time = time.time() - start_time
        self.logger.info('Testing time: %.3f' % self.test_time)
        test_labels = torch.cat(test_labels,dim=0)
        test_labels_NF=test_labels[:,0].squeeze().cpu().detach().numpy()
        f1 = f1_score(test_labels_NF, preds, average='binary')
        accuracy = accuracy_score(test_labels_NF, preds)
        precision = precision_score(test_labels_NF, preds, average='binary')
        recall = recall_score(test_labels_NF, preds, average='binary')
        self.logger.info('Anomaly Detection Accuracy: ' + str(accuracy))
        self.logger.info('Anomaly Detection Precision: ' + str(precision))
        self.logger.info('Anomaly Detection Recall: ' + str(recall))
        self.logger.info('Anomaly Detection F1: ' + str(f1))
        self.logger.info('Finished testing.')

    def vae_loss_cal(self, ori_hg, ori_hg_hat, ori_mu, ori_logvar, neg_hg, neg_hg_hat, neg_mu,neg_logvar):
        ori_sigma = torch.exp(0.5 * ori_logvar)
        neg_sigma = torch.exp(0.5 * neg_logvar)
        diff = ori_hg-ori_hg_hat
        sq_diff=torch.square(diff)
        ori_reconst_loss = sq_diff.mean()
        ori_kl_div = 0.5 * torch.sum(ori_mu ** 2 + ori_sigma ** 2 - torch.log(1e-8 + ori_sigma ** 2) - 1,dim=1)
        ori_kl_div_mean=ori_kl_div.mean()
        ori_loss = ori_reconst_loss + ori_kl_div_mean
        neg_diff = neg_hg - neg_hg_hat
        neg_sq_diff = torch.square(neg_diff)
        neg_reconst_loss  = neg_sq_diff.mean()
        neg_kl_div = 0.5 * torch.sum(neg_mu**2+neg_sigma**2-torch.log(1e-8+neg_sigma**2)-1,dim=1)
        neg_kl_div_mean =neg_kl_div.mean()
        neg_loss = neg_reconst_loss+neg_kl_div_mean
        all_dis = self.compute_kl(ori_mu.mean(dim=0).unsqueeze(dim=0),ori_sigma.mean(dim=0).unsqueeze(dim=0),
                                  neg_mu.mean(dim=0).unsqueeze(dim=0),neg_sigma.mean(dim=0).unsqueeze(dim=0))
        loss = (ori_loss + neg_loss) / (self.beta * all_dis).clamp_(min=1e-8)
        if torch.isnan(loss):
            loss = torch.tensor(1e30,requires_grad=True)
        return loss

    def compute_kl(self,mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None):
        eps=10**-8
        sigma_poster = sigma_poster ** 2
        sigma_prior = sigma_prior ** 2
        sigma_poster_matrix_det = torch.prod(sigma_poster, dim=1)
        sigma_prior_matrix_det = torch.prod(sigma_prior, dim=1)
        sigma_prior_matrix_inv = 1.0 / (sigma_prior+eps)
        delta_u = (mu_prior - mu_poster)
        term1 = torch.sum(sigma_poster / (sigma_prior+eps), dim=1)
        term2 = torch.sum(delta_u * sigma_prior_matrix_inv * delta_u, 1)
        term3 = - mu_poster.shape[-1]
        term4 = torch.log(sigma_prior_matrix_det + eps) - torch.log(sigma_poster_matrix_det + eps)
        kl_loss = 0.5 * (term1 + term2 + term3 + term4)
        kl_loss = torch.clamp(kl_loss, 0, 100)
        return torch.mean(kl_loss)

    def GaussianMixture_model(self,train_normal_embeddings, train_anomaly_embeddings):
        K = self.k
        normal_gmm = GaussianMixture(n_components=K, covariance_type='full', max_iter=100, random_state=42)
        normal_gmm.fit(train_normal_embeddings)
        anomaly_gmm = GaussianMixture(n_components=K, covariance_type='full', max_iter=100, random_state=42)
        anomaly_gmm.fit(train_anomaly_embeddings)
        return normal_gmm,anomaly_gmm

