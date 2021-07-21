# VAE-SVDD
A VAE SVDD One Class Classification (For MNIST) 97-98% F1 with a simple 2 layer CNN
## One Class Classification on Data Stream

##### Experiment	VAE-SVDD on batch MNIST

MNIST 1 Vesus Other (60000for Training ,10000 for testing)： 

| Dataset | AS in P | AS in N | F1 score | Time(s) |
| :-----: | :-----: | :-----: | :------: | :-----: |
|    0    | 0.999😘  |  0.977  |  0.986   |  33.2   |
|    1    | 1.000😍  |  0.986  |  0.993   |  34.7   |
|    2    |  0.993  |  0.972  |  0.959   |  30.4   |
|    3    |  0.995  |  0.983  |  0.971   |  29.8   |
|    4    |  0.998  |  0.976  |  0.981   |  27.8   |
|    5    |  0.998  |  0.974  |  0.980   |  30.3   |
|    6    |  0.999  |  0.978  |  0.985   |  31.0   |
|    7    |  0.998  |  0.984  |  0.987   |  32.5   |
|    8    |  0.981  |  0.978  |  0.901😥  |  30.1   |
|    9    |  0.966  |  0.970  |  0.928🥲  |  30.7   |

**EVN**:Win 11 dev😤, torch 1.9.0 stable, cuda 11.0 , scikit learning 0.24.2, python 3.9.0

​		3700x 8 cores with 4.0GHz, 2*8G RAM on 2133 Mhz, RTX 2080 defualt, Nvme SSD😅

**Network**:(All NN inplemented pyTorch) 

[![WdCRN6.png](https://z3.ax1x.com/2021/07/21/WdCRN6.png)](https://imgtu.com/i/WdCRN6)

Graph made by [NN SVG (alexlenail.me)](http://alexlenail.me/NN-SVG/AlexNet.html) and Draw.IO🤞

**SVDD**:  nu=0.008, RBF kernerl (implemented by sci-kit learn) and other use defualt setting.

