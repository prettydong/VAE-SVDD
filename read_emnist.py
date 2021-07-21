import numpy
from torchvision import datasets, transforms
import pickle


# 迭代器的第一项是 迭代器所在的内存地址
# print(next(i)) #第一次使用next 才能得到迭代器的第一个元素
# 该迭代器中，每个元素有两项，第一项是一个tensor第二项 是他的标签

# print(len(emnist_dataset)) # 迭代器似乎可以通过len去获得其中元素的个数
# 上述说法不正确，通过len可以获得长度是因为，对象有__len__的属性，和其是不是迭代器对象无关

# 迭代器的写法。使用try尝试，否则结束是会报错退出
# print(emnist_dataset.targets)
# b = numpy.unique(emnist_dataset.targets.numpy())
# c = (b.tolist())
# print(c)
# 获取所有的标签的列表
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
# 11, 12, 13, 14, 15, 16, 17, 18, 19,
# 20, 21, 22, 23, 24, 25, 26, 27, 28,
# 29, 30, 31, 32, 33, 34, 35, 36, 37,
# 38, 39, 40, 41, 42, 43, 44, 45, 46]
# k = 0
# 每个类有2400个实例，总计2400*47 = 112800个实例

class DataReader:
    """
    only for EMNIST
    """

    def __init__(self):
        self.label_num = 2000
        self.unlabel_num = 400
        self.emnist_dataset = datasets.MNIST('data', train=True, download=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor()
                                             ]))
        self.data_with_label = [[] for i in range(47)]
        self.data_without_label = [[] for i in range(47)]
        self.target_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                            32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                            42, 43, 44, 45, 46]
        self.label_cnt = [0 for i in range(47)]
        # 表示这个位置的label已经被抽出了多少次
        self.target_label_num = [self.label_num for i in range(47)]
        self.target_unlabel_num = [self.unlabel_num for i in range(47)]
        # 先选出的labeled的数据，然后再选出unlabel的数据

    def clean(self):
        self.data_with_label = []

    def extract(self, chosen_list):
        def one_hot(k):
            l = len(chosen_list)
            one_hot_code = [0 for _ in range(l)]
            one_hot_code[k] = 1
            return one_hot_code

        data_i = iter(self.emnist_dataset)
        for i in chosen_list:
            if i not in self.target_list:
                raise NameError('chosen class not in target list! please check!')
        while True:
            try:
                p = next(data_i)
                if p[1] in chosen_list:  # 1的位置是target
                    if self.label_cnt[p[1]]<self.label_num:
                        self.label_cnt[p[1]] += 1
                        temp = (p[0], one_hot(p[1]))
                        self.data_with_label[p[1]].append(p)
                    elif self.label_cnt[p[1]]<self.label_num + self.unlabel_num:
                        self.label_cnt[p[1]] += 1
                        temp = (p[0], one_hot(p[1]))
                        self.data_without_label[p[1]].append(p)
            except StopIteration:
                break

    def show(self):
        print(self.data_with_label)
        print(self.data_without_label)

    def get_a_class(self, n):
        return self.data_with_label[n],self.data_without_label[n]

    def get_all_classes(self):
        label_ret = []
        for i in self.data_with_label:
            label_ret = label_ret + i
        unlabel_ret = []
        for i in self.data_without_label:
            unlabel_ret = unlabel_ret + i
        return label_ret,unlabel_ret

    def save_all_as_pkl(self, path):
        f = open(path, "wb")
        pickle.dump(self.get_all_classes(), f)
        f.close()


if __name__ == '__main__':
    data_reader = DataReader()
    # use select_random_class.py to get n random classes: [10, 5, 24, 25, 22, 18, 19, 13, 37, 35]
    # and we use those class to construct a base classifier
    # chosen_list = [10, 5, 24, 25, 22, 18, 19, 13, 37, 35]
    chosen_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_reader.extract(chosen_list)
    print(len(data_reader.get_all_classes()))
    data_reader.save_all_as_pkl("./data/emnistpkl/base_classes.pkl")
