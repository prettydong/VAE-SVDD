import time

from sklearn.datasets import load_digits
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score

# OC-SVM in sklearn is SVDD

data = load_digits().data
target = load_digits().target

t = zip(data, target)

ONE_CLASS_LABEL = [8]
OTHER_CLASS_LABEL = [1, 2, 3, 4, 5, 6, 8, 9, 0]

occ_data, occ_target = [], []
other_data, other_target = [], []
for i in t:
    if i[1] in ONE_CLASS_LABEL:
        occ_data.append(i[0])
        occ_target.append(i[1])
    elif i[1] in OTHER_CLASS_LABEL:
        other_data.append(i[0])
        other_target.append(i[1])

one_class_classifier = OneClassSVM(kernel="rbf",nu=0.05)
one_class_classifier.fit(occ_data)

pred = one_class_classifier.predict(occ_data)
pred_ = one_class_classifier.predict(other_data)
right = [1 for i in range(len(occ_data))]
right_ = [-1 for i in range(len(other_data))]
print(accuracy_score(right, pred))
print(accuracy_score(right_,pred_))
print(time.asctime(time.localtime(time.time())))
