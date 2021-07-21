import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F
from read_mnist import get_one_class_from_mnist


class VAE(nn.Module):
    def __init__(self, kld_weight):
        super().__init__()
        self.kld_weight = kld_weight
        self.leakyRelu = nn.LeakyReLU()

        dims1 = 16
        dims2 = 4
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=dims1, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(dims1)
        self.cnn2 = nn.Conv2d(in_channels=dims1, out_channels=dims2, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(dims2)

        self.fc_var = nn.Linear(1600, 32)
        self.fc_mu = nn.Linear(1600, 32)

        self.imu_test = nn.Linear(32, 1600)

        self.icnn2 = nn.ConvTranspose2d(in_channels=dims2, out_channels=dims1, kernel_size=5)
        self.icnn1 = nn.ConvTranspose2d(in_channels=dims1, out_channels=1, kernel_size=5)

    def encode2latent_dims(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.leakyRelu(x)
        x = torch.flatten(x, start_dim=1)

        return x

    def hidden_dims2mu_var(self, x):
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

    def repara(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        z = self.encode2latent_dims(x)
        mu, var = self.hidden_dims2mu_var(z)
        z = self.repara(mu, var)
        pred_x = self.decode_from_latent_dims(z)
        return pred_x, mu, var

    def decode_from_latent_dims(self, x):
        # x = self.imu_test(x)
        x = self.imu_test(x)
        x = x.view(-1, 4, 20, 20)
        x = self.icnn2(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)
        x = self.icnn1(x)
        # x = self.bn2(x)
        x = self.leakyRelu(x)
        return x

    def loss(self, recons_x, origin_x, var, mu):
        recons_loss = F.mse_loss(recons_x, origin_x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)
        c = torch.mean(mu)
        r = mu - c
        r = torch.sum(r*r)
        loss_ = recons_loss + self.kld_weight * kld_loss
        # print(recons_loss,kld_loss)
        return loss_


if __name__ == '__main__':
    t = get_one_class_from_mnist()
    w = t["one_class_data"][0] / 256
    w = torch.unsqueeze(w, 0)
    w = torch.unsqueeze(w, 0)
    y = VAE()
    g = None
    for i in range(1000):
        g = y.encode2latent_dims(w)
        g = y.decode_from_latent_dims(g)[0][0]
        opt = torch.optim.Adam(y.parameters(), lr=1e-3)
        loss = y.loss(w[0][0], g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
