from __future__ import division,print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.data_operation import accuracy_score
from utils.data_manipulation import train_test_split,normalize
from utils.kernels import *
from supervised_learning.support_vector_machine import SuperVectorMachine
def main():
    data  = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y==1] = -1
    y[y==2] = 1

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

    clf = SuperVectorMachine(kernel=ploynomial_kernal,power=4,coef=1)

    clf.fit(X_train,y_train)
    y_pred  = clf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    print("Accuracy :" ,accuracy)
    print(X_test.shape)

if __name__ == '__main__':
    main()