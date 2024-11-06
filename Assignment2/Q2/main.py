import numpy as np
import matplotlib.pyplot as plt
from dataset import upload_csv
from analytical import w_analytic
from gradient_descent import gradient_descent, GD_plot, L2_norm
from SGD import SGD, SGD_plot
from ridge_regression import ridge_regression, RidgeRegression_plot, loss_vs_lambda_plot


dataset = upload_csv('/home/ruthwik/Sem 6/PRML/assignment2/A2Q2Data_train.csv')

X = dataset[:,:-1].T
y = dataset[:,-1].T
w_ML = w_analytic(X,y)

GD_plot(X, y, 1e-6, 5000)
SGD_plot(X, y, 1e-4, 5000)
train_loss = RidgeRegression_plot(X, y, 1e+4, 1e-6, 5000)
print(train_loss[-1])
loss_vs_lambda_plot(X, y, [1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5], 1e-6, 100)
wR, _, _ = ridge_regression(X, y, 1e+4, 1e-6, 5000)

test_data = upload_csv('/home/ruthwik/Sem 6/PRML/assignment2/A2Q2Data_test.csv')
x_test = test_data[:,:-1].T
y_test = test_data[:,-1].T

wML_loss = L2_norm(x_test.T@w_ML - y_test)
wR_loss = L2_norm(x_test.T@wR - y_test)

print("analytical loss: ",wML_loss)
print("ridge regression loss: ",wR_loss)

