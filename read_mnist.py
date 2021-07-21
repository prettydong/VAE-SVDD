import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def get_one_class_from_mnist(one_class_list=None,train=True):
    if one_class_list is None:
        one_class_list = [0]
    mnist_dataset = datasets.MNIST('data', train=train, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ]))
    data = mnist_dataset.data
    target = mnist_dataset.targets

    one_class_data = []
    outlier_data = []
    is_outlier = []
    for data_x, label_x in zip(data, target):
        if label_x in one_class_list:
            w = data_x / 256
            w = torch.unsqueeze(w, 0)
            one_class_data.append(w)
            is_outlier.append(0)
        else:
            w = data_x / 256
            w = torch.unsqueeze(w, 0)
            outlier_data.append(w)
            is_outlier.append(1)

    mnist_dict = {
        "one_class_data": one_class_data,
        "outlier_data": outlier_data,
        "is_outlier": is_outlier
    }
    return mnist_dict


if __name__ == '__main__':
    t = get_one_class_from_mnist(train=False)
    print(t["one_class_data"][0])
    w = t["one_class_data"][0]
