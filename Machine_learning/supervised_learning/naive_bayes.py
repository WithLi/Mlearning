from __future__ import division,print_function

import numpy as np
import math

from utils.data_manipulation import train_test_split,normalize
from utils.misc import Plot
from utils.data_operation import accuracy_score

class NaiveBayes():
    """The Gaussion Navie Bayes classifier"""
    """高斯朴素贝叶斯"""
    def fit(self,X,y):
        self.X,self.y = X,y
        self.classes = np.unique(y)
        self.parameters = []
        print(X.shape)
        #Calculate the mean and variance of each feature for each class
        for i,c in enumerate(self.classes):
            #Only select the rows where the label equal the given class
            X_where_c = X[np.where(y == c)]
            print(X_where_c.shape)
            self.parameters.append([])
            #Add the mean and variance for each features(column)
            for j in range(X.shape[1]):
                """获取每一列的值"""
                col = X_where_c[:,j]
                """获取每一列的平均值和方差"""
                parameters = {"mean":col.mean(),"var":col.var()}
                self.parameters[i].append(parameters)

    def _calculate_likelihood(self,mean,var,x):
        """Gaussion likelihood of the data x given mean and var"""
        """根据期望，方差，计算高斯分布"""
        eps = 1e-4 #add in denominator to prevent division by zero
        coeff = 1.0/ math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean,2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self,c):
        """Calculcate the prior of class c
        (samples where class == c / total number of samples)
        """
        """计算C 的先验概率"""
        X_where_c = self.X[np.where(self.y == c)]
        n_class_instances = X_where_c.shape[0] #计算所以为Y= C的个数
        n_total_instances = self.X.shape[0] #计算y的所有的个数

        return n_class_instances/n_total_instances

    def _classify(self,sample):
        """ Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X)
                P(X|Y) - Likelihood of data X given class distribution Y.
                         Gaussian distribution (given by _calculate_likelihood)
                P(Y)   - Prior (given by _calculate_prior)
                P(X)   - Scales the posterior to make it a proper probability distribution.
                         This term is ignored in this implementation since it doesn't affect
                         which class distribution the sample is most likely to belong to.
                Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = []
        #Go throght list of classes
        for i,c in enumerate(self.classes):
            """获取先验概率"""
            posterior = self._calculate_prior(c)
            #Naive assumption (independence)
            #P(x1,x2,x3|y) = p(x1|y)*p(x2|y)p(x3|y)
            #Mulitipy with the class likeihoods
            """:parameter 有多少个特征"""
            for j,params in enumerate(self.parameters[i]):
                """sample_features 为第几个特征"""
                sample_features = sample[j]
                #Determine p(X|y)
                likelihood = self._calculate_likelihood(params["mean"],params["var"],sample_features)
                #Muliply with the accumulated probability
                posterior *= likelihood
            #Total posteriors = p(x1|y)*p(x2|y)p(x3|y)*p(Y)
            posteriors.append(posterior)
        #return the class with the largest posterior probability
        index_of_max = np.argmax(posteriors)
        return self.classes[index_of_max]

    def predict(self,X):
        """Predict the class labels of the samples in X"""
        print(X.shape)
        y_pred = []
        for sample in X:
            y = self._classify(sample)
            y_pred.append(y)

        return y_pred



