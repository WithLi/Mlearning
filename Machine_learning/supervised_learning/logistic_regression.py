import math

import numpy as np

from utils.function import Sigmoid


class LogisticRegression():
    """ Logistic Regression classifier.
        Parameters:
        -----------
        learning_rate: float
            The step length that will be taken when following the negative gradient during
            training.
        gradient_descent: boolean
            True or false depending if gradient descent should be used when training. If
            false then we use batch optimization by least squares.
    """
    def __init__(self,learning_rate=.1,gradient_descent = True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self,X):
        n_features = np.shape(X)[1]   #获取特征的个数
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1/math.sqrt(n_features)
        self.param = np.random.uniform(-limit,limit,(n_features))

    def fit(self,X,y,n_iterations = 4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            #make a new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            #执行梯度下降
            self.param = self.param - self.learning_rate * -(y - y_pred).dot(X)


    #预测
    def predict(self,X):
        y_pred = np.round(self.sigmoid(X.dot(self.param)))
        return y_pred.astype(int)


