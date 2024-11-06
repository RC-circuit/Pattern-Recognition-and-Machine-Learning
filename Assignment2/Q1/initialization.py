import numpy as np
import random as rand


def bernoulli_initializer(x,clusters):
    p = [[rand.random() for i in range(x.shape[1])] for j in range(clusters)]
    pi = [rand.random() for i in range(clusters)]
    s = sum(pi)
    pi = [ i/s for i in pi ]

    return p, pi

def gaussian_initializer(x,clusters):
    mu = [[rand.random() for i in range(x.shape[1])] for j in range(clusters)]
    pi = [rand.random() for i in range(clusters)]
    s = sum(pi)
    pi = [ i/s for i in pi ]
    cov = np.random.rand(clusters, x.shape[1], x.shape[1])
    return np.array(mu), cov, np.array(pi)