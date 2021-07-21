import os

import torch
from torch.utils.data import DataLoader

from AE import AE
from read_mnist import get_one_class_from_mnist
from utils import *
from sklearn.svm import OneClassSVM


def load_ae():
    print(torch.__version__)
    # print(torch.cuda.is_available())
    device = torch.device('cuda:0')
    model = AE()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    if os.path.exists("ae_checkpoint.dict"):
        checkpoint = torch.load("ae_checkpoint.dict")
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, opt


def training_ae():
    epoch = 100
    device = torch.device('cuda:0')
    t = get_one_class_from_mnist()
    w = t["one_class_data"]
    trainloader = DataLoader(w, batch_size=256)
    model, opt = load_ae()
    if os.path.exists("ae_checkpoint.dict"):
        checkpoint = torch.load("ae_checkpoint.dict")
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    for i in range(epoch):
        print('epoch:', i + 1)
        avg_loss = []
        for data in trainloader:
            data = data.to(device)
            z = model.encode2latent_dims(data)
            pred_y = model.decode_from_latent_dims(z)
            loss = model.loss(data, pred_y)
            avg_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(sum(avg_loss) / len(avg_loss))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
    }, "ae_checkpoint.dict")


def test_ae(n: int):
    model, x = load_ae()
    device = torch.device('cuda:0')
    t = get_one_class_from_mnist()
    for i in range(n):
        w = t["outlier_data"][i] / 256
        w = torch.unsqueeze(w, 0)
        w = w.to(device)
        z = model.encode2latent_dims(w)
        pred_y = model.decode_from_latent_dims(z)
        draw(pred_y[0][0], "gen_img/test" + str(i))


def reduction(data):
    model, x = load_ae()
    device = torch.device('cuda:0')
    ret_z = []
    for i in data:
        w = i / 256
        w = torch.unsqueeze(w, 0)
        w = w.to(device)
        z = model.encode2latent_dims(w)
        z = z[0].to('cpu').detach().numpy()
        ret_z.append(z)
    return ret_z


if __name__ == '__main__':
    test_ae(10)
    # t = get_one_class_from_mnist()
    # data_one = t["one_class_data"]
    # data_one_z = reduction(data_one)
    # data_outlier = t["outlier_data"]
    # data_outlier_z = reduction(data_outlier)
    # svdd = OneClassSVM(nu=0.03)
    # svdd.fit(data_one_z)
    # pred_o = svdd.predict(data_outlier_z)
    # pred_in = svdd.predict(data_one_z)
    # sum_correct = 0
    # for i in pred_o:
    #     if i==-1 :
    #         sum_correct = sum_correct+1
    # print(sum_correct/len(pred_o))
    # sum_correct = 0
    # for i in pred_in:
    #     if i== 1 :
    #         sum_correct = sum_correct+1
    # print(sum_correct/len(pred_in))
