import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F
from read_mnist import get_one_class_from_mnist


def draw(t, name="mygraph.png"):
    t = t.cpu().detach().numpy()
    plt.imshow(t)
    plt.savefig(name)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyRelu = nn.LeakyReLU()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(8)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(4)

        self.fc_var = nn.Linear(1600, 32)
        self.fc_mu = nn.Linear(1600, 32)

        self.imu_test = nn.Linear(32, 1600)

        self.icnn2 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=5)
        self.icnn1 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5)

    def encode2latent_dims(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.leakyRelu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_mu(x)
        return x

    def hidden_dims2mu_var(self, x):
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

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

    def loss(self, recons_x, origin_x):
        recons_loss = F.mse_loss(recons_x, origin_x)
        return recons_loss


if __name__ == '__main__':
    t = get_one_class_from_mnist()
    w = t["one_class_data"][0] / 256
    w = torch.unsqueeze(w, 0)
    w = torch.unsqueeze(w, 0)
    y = AE()
    g = None
    for i in range(1000):
        g = y.encode2latent_dims(w)
        g = y.decode_from_latent_dims(g)[0][0]
        loss = y.loss(w[0][0], g)
        print(loss)
