import numpy as np
import math
import sys


def accuracy_score(y_true,y_pred):
    """Compare y_true to y_pred and return accuracy"""
    accuracy = np.sum(y_true == y_pred,axis=0) / len(y_true)
    return accuracy

def calculate_covariance_matrix(X,Y=None):
    """Calculate the convariance matrix for the dataset X"""
    """计算协方差矩阵"""
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis= 0))
    return np.array(covariance_matrix,dtype=float)
def calculate_variance(X):
    """:return the variane of the features in dataset X"""
    """方差"""
    mean = np.ones(np.shape(X))*X.mean(0)
    n_samples = np.shape(X)[0]
    variance  = (1 / n_samples) * np.diag((X-mean).T.dot(X - mean))
    return variance

def calculate_std_dev(X):
    """Calculate the standard deviations of the features in datasets"""
    """ 标准差 """
    std_dev = np.sqrt(calculate_variance(X))
    return  std_dev

def calculate_correlation_matrix(X,Y=None):
    """Calculate the correlation matrix for the dataset X"""
    """相关系数矩阵"""
    if Y is None:
        Y = X
    n_samples  = np.shape(X)[0]
    covariance = (1 / n_samples)*(X-X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X),1)
    std_dev_y = np.expand_dims(calculate_std_dev(X),1)
    correlation_matrix = np.divide(covariance,std_dev_X.dot(std_dev_y.T))
    return np.array(correlation_matrix,dtype=float)

def mean_squared_error(y_true,y_pred):
    """Return the mean squared error betweed y_true and ty_pred"""
    mse = np.mean(np.power(y_true - y_pred,2))
    return mse

def euclidean_distance(x1,x2):
    """Calculates the l2 distance betweed two vectors"""
    distance = 0
    #squared distance between each coordinate
    for i in range(len(x1)):
        distance +=pow((x1[i] - x2[i]),2)
    return math.sqrt(distance)
