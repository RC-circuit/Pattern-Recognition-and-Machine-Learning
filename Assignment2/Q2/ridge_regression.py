import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import L2_norm


def ridge_regression(X, Y, l, lr, iterations):
    train_loss = []
    w = np.random.rand(100,)
    indices = np.random.randint(0,10000, size = 7000)
    x = X[:,indices]
    y = Y[indices]
    for t in range(iterations):
        grad_f = 2*((x@x.T)@w - x@y + l*w)
        w = w - (lr)*grad_f
        train_loss.append(L2_norm(x.T@w - y))

    validation_indices = [i for i in range(10000) if i not in indices]
    x_valid = X[:,validation_indices]
    y_valid = Y[validation_indices]
    valid_loss = L2_norm(x_valid.T@w - y_valid)
    return w, train_loss, valid_loss

def RidgeRegression_plot(X, y, l, lr, iterations):
    _, train_loss, _ = ridge_regression(X, y, l, lr, iterations)
    plt.plot([t+1 for t in range(iterations)],train_loss)
    plt.xlabel('iterations')
    plt.ylabel('Loss function ')
    plt.title('Ridge Regression')
    plt.show()
    return train_loss

def loss_vs_lambda_plot(X, y, lambdas, lr, iterations):
    loss = []
    for l in lambdas:
        _, _, valid_loss = ridge_regression(X, y, l, lr, iterations)
        loss.append(valid_loss)
    plt.plot(lambdas,loss)
    plt.xlabel('lambda')
    plt.ylabel('validation loss')
    plt.title('Validation Loss vs Lambda')
    plt.xticks()
    plt.show()
