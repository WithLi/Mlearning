import numpy as np

def linear_kernel(**kwargs):
    def f(x1,x2):
        return np.inner(x1,x2)
    return f

def ploynomial_kernal(power,coef,**kwargs):
    def f(x1,x2):
        return (np.inner(x1,x2) +coef)**power
    return f

def rbf_kernal(gamma,**kwargs):
    def f(x1,x2):
        distance = np.linalg.norm(x1-x2)**2
        return np.exp(-gamma * distance)