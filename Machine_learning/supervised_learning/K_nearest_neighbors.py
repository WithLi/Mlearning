from __future__ import print_function,division
import numpy as np
from utils.data_operation import euclidean_distance

class KNN():
    """ K Nearest Neighbors classifier.
    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.
    """
    def __init__(self,k=5):
        self.k = k

    def _vote(self,neighbors):
        """:return the common class among the neighbor sample"""
        counts = np.bincount(neighbors[:,1].astype('int'))
        return counts.argmax()

    def predict(self,X_test,X_train,y_train):
        y_pred = np.empty(X_test.shape[0])
        #determine the class of each sample
        for i,test_sample in enumerate(X_test):
            #two columns [distance ,label] for each observed sample
            '''产生一个二维数组'''
            neighbor = np.empty((X_train.shape[0],2))
            # Calculate the distance from each observed sample to the
            # sample we wish to predict
            for j,observed_sample in enumerate(X_train):
                distance = euclidean_distance(test_sample,observed_sample)
                label = y_train[j]
                #add neighbor information
                neighbor[j] = [distance,label]
            #sort the list of observed sample from lowest to highest distance
            #and select the K first
            k_nearest_neighbors = neighbor[neighbor[:,0].argsort()][:self.k]
            #get the most common class among the neigbors
            label  = self._vote(k_nearest_neighbors)
            y_pred[i] = label
        return y_pred
