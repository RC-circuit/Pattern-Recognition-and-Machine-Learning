import numpy as np
import matplotlib.pyplot as plt
from analytical import w_analytic

def L2_norm(x):
    return np.sqrt(np.sum(np.square(x)))

def gradient_descent(X, y, lr, iterations):
    norm = []
    w_ML = w_analytic(X,y)
    w = np.random.rand(100,)
    #norm.append(L2_norm(w - w_ML))

    for t in range(iterations):
        grad_f = 2*(X@X.T)@w - 2*X@y
        w = w - (lr)*grad_f
        norm.append(L2_norm(w - w_ML))

    return w, norm

def GD_plot(X, y, lr, iterations):
    w_GD, norm = gradient_descent(X, y, lr, iterations)
    plt.plot([t+1 for t in range(iterations)],norm)
    plt.xlabel('iterations')
    plt.ylabel('L2 norm: ||w - w_ML|| ')
    plt.title('Gradient Descent')
    plt.show()



