import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import L2_norm
from analytical import w_analytic

def SGD(X, Y, lr, iterations):
    norm = []
    w_ML = w_analytic(X,Y)
    w = np.random.rand(100,)
    #norm.append(L2_norm(w - w_ML))

    for t in range(iterations):
        indices = np.random.randint(0,10000, size = 100)
        x = X[:,indices]
        y = Y[indices]
        grad_f = 2*(x@x.T)@w - 2*x@y
        w = w - (lr)*grad_f
        norm.append(L2_norm(w - w_ML))

    return w, norm

def SGD_plot(X, y, lr, iterations):
    w_SGD, norm = SGD(X, y, lr, iterations)
    plt.plot([t+1 for t in range(iterations)],norm)
    plt.xlabel('iterations')
    plt.ylabel('L2 norm: ||w - w_ML|| ')
    plt.title('Stochastic Gradient Descent')
    plt.show()
