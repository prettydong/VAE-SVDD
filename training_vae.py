import os

import torch
from torch.utils.data import DataLoader
import numpy
from baseVAE import VAE
from read_mnist import get_one_class_from_mnist
from utils import *
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



def load_vae(batch_size, load=False):
    print(torch.__version__)
    # print(torch.cuda.is_available())
    device = torch.device('cuda:0')
    model = VAE(batch_size / 6000)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    # opt = torch.optim.SGD(model.parameters(),lr=1e-4)
    # opt = torch.optim.RMSprop(model.parameters(),lr = 1e-4)
    if os.path.exists("vae_checkpoint.dict") and load == True:
        checkpoint = torch.load("vae_checkpoint.dict")
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, opt


def training_vae(batch_size=256, epoch=200, training_list=[0]):
    device = torch.device('cuda:0')
    t = get_one_class_from_mnist(training_list)
    w = t["one_class_data"]
    trainloader = DataLoader(w, batch_size)
    model, opt = load_vae(batch_size)
    for i in range(epoch):

        avg_loss = []
        for data in trainloader:
            data = data.to(device)
            # z = model.encode2latent_dims(data)
            # pred_y = model.decode_from_latent_dims(z)
            # loss = model.loss(data, pred_y)
            pred_x, mu, var = model.forward(data)
            loss = model.loss(data, pred_x, var, mu)
            avg_loss.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        if i % 100 == 0:
            print('epoch:', i + 1)
            print(sum(avg_loss) / len(avg_loss))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
    }, "vae_checkpoint.dict")


def test_vae(n: int):
    model, x = load_vae(256)
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
    model, x = load_vae(256)
    device = torch.device('cuda:0')
    testloader = DataLoader(data, 256)
    init_flag = False
    ret_z = "may be something wrong"
    for i in testloader:
        w = i / 256
        w = w.to(device)
        z = model.encode2latent_dims(w)
        z = model.hidden_dims2mu_var(z)[0]
        z = z.to('cpu').detach().numpy()

        if init_flag:
            ret_z = numpy.vstack((ret_z, z))
        else:
            ret_z = z
            init_flag = True
    return ret_z


def test_on_mnist(one_class_list):
    training_vae(batch_size=512, epoch=200, training_list=one_class_list)
    t = get_one_class_from_mnist(one_class_list, train=False)
    data_one = t["one_class_data"]
    data_one_z = reduction(data_one)

    data_outlier = t["outlier_data"]
    data_outlier_z = reduction(data_outlier)

    print("SVDD:" + str(0.008))
    svdd = OneClassSVM(nu=0.008, kernel="rbf")
    svdd.fit(data_one_z)
    pred_o = svdd.predict(data_outlier_z)
    pred_in = svdd.predict(data_one_z)
    pred_all = numpy.hstack((pred_o, pred_in))
    ans1 = [-1 for i in pred_o]
    ans2 = [1 for i in pred_in]
    ans_all = ans1 + ans2
    print("the One class is", one_class_list)
    print(accuracy_score(pred_o, ans1))
    print(accuracy_score(pred_in, ans2))
    print(f1_score(pred_all, ans_all))
    print("")


if __name__ == '__main__':
    time_recorder = TimeRecorder()
    for i in range(10):
        time_recorder.start_record()
        _ = [i]
        test_on_mnist(_)
        print("Time Using:",time_recorder.end_record_return())