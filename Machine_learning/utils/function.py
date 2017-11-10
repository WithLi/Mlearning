import numpy as np

class Sigmoid():
    def __call__(self, x):
        return 1 / (1+np.exp(-x))   #sigmoid 函数
    '''
     gradient 为sigmoid 的导数，sigmoid导数为f(x)' = f(x)*(1-f(x))
    '''
    def gradient(self,x):
        return self.__call__(x)*(1-self.__call__(x))