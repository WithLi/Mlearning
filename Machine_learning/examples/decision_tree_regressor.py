from __future__ import division,print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_manipulation import train_test_split,standardize
from utils.data_operation import accuracy_score,mean_squared_error,calculate_variance
from supervised_learning.decision_tree import RegressionTree

def main():

    print("---Regression Tree----")

    #load temperature data
    data = pd.read_csv('../data/TempLinkoping2016.txt',sep="\t")

    time = np.atleast_2d(data["time"].as_matrix()).T
    temp = np.atleast_2d(data["temp"].as_matrix()).T

    X = standardize(time)
    y = temp[:,0]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    model = RegressionTree()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)


if __name__ == '__main__':
    main()