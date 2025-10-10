import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
import torch.nn.init as init

class VAE(nn.Module):
    def __init__(self, out_channels):
        super(VAE, self).__init__()
        self.out_channels=out_channels
        self.double_channels=int(2*self.out_channels)
        self.div_channels=int(self.out_channels/2)
        self.num_samples = 10

        self.fc1 = nn.Linear(self.out_channels, 128)
        self.bn1 = BatchNorm(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = BatchNorm(256)
        self.fc21 = nn.Linear(256, self.div_channels)
        self.fc22 = nn.Linear(256, self.div_channels)

        self.fc3 = nn.Linear(self.div_channels, 256)
        self.bn3 = BatchNorm(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = BatchNorm(128)
        self.fc5 = nn.Linear(128, self.out_channels)
        self.bn5 = BatchNorm(self.out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def encoder(self, x):
        h = self.bn1(torch.relu(self.fc1(x)))
        h = self.bn2(torch.relu(self.fc2(h)))
        return self.fc21(h), self.fc22(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn((self.num_samples, *mu.shape)).to(mu.device)
        z = mu + eps * std
        return z

    def decoder(self, z):
        z = z.view(-1, z.size(-1))
        h = self.bn3(torch.relu(self.fc3(z)))
        h = self.bn4(torch.relu(self.fc4(h)))
        h = h.view(self.num_samples, -1, h.size(-1))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.out_channels))
        h_=torch.cat((mu,log_var),dim=1)
        z = self.sampling(mu, log_var)
        x_hat=self.decoder(z)
        x_hat=x_hat.view(self.num_samples, -1, self.out_channels).mean(dim=0)
        return h_,x_hat, mu, log_var



