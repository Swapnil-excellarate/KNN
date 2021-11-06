import numpy as np
from collections import Counter


def accuracy_score(y_pred, y_true):
    accuracy=np.sum(y_true == y_pred)/ len(y_true)
    return accuracy

def eclD(p, q):
    ecld=np.sqrt(sum(p-q)**2)
    return ecld


class knn:
    def __init__(self, k):
        self.k=k

    def fit(self, X, y):
        self.x_train=X
        self.y_train=y

    def predict(self, X):
        predict_result=[self.predictDef(x) for x in X]
        return predict_result

    def predictDef(self, x):
        distance=[eclD(x, x1) for x1 in self.x_train]
        nearestNeighber=np.argsort(distance)[:self.k]
        nearestNeighber_y=[self.y_train[i] for i in nearestNeighber]
        label=(Counter(nearestNeighber_y).most_common())[0][0]
        return label


