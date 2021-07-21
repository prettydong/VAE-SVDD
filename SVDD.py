import time
from sklearn.datasets import load_digits
from sklearn.svm import OneClassSVM


class SVDD:
    def __init__(self, **kwargs):
        """
        :param kwargs:
        data -> 2parts: 1.data 2. weight:real==1 , pseudo: to be set
        """
        data = kwargs["data"]
        self.one_class_classifier = OneClassSVM(**kwargs)
        self.create_svdd(data)

    def create_svdd(self, data):
        self.one_class_classifier.fit(data[0], sample_weight=data[1])

    def predict_point(self,x):
        return self.one_class_classifier.predict([x])

    def predict_list(self,x):
        return self.one_class_classifier.predict(x)

