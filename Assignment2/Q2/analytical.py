import numpy as np
from numpy.linalg import inv

def w_analytic(X,y):
    return inv(X@X.T)@X@y
