import numpy as np
import matplotlib.pyplot as plt
from PCA import find_topK_eigenpairs
import math


def polynomial_kernel(x1, x2, d):
    return ((x1.T)@(x2) + 1)**d

def radial_basis_kernel(x1, x2, sigma):
    return math.exp( (-1*((x1 - x2).T)@(x1 - x2))/ (2*(sigma**2)) )

def centered_K_matrix(X, kernel_func, param):
    
    K = np.empty([X.shape[1], X.shape[1]])

    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            K[i,j] = kernel_func(X[:,i],X[:,j],param)

    for i in range(X.shape[1]):
        K_i = 0
        for j in range(X.shape[1]):
            K_i += kernel_func(X[:,i],X[:,j],param)
        K[i,:] -= K_i/X.shape[1]

    for j in range(X.shape[1]):
        K_j = 0
        for i in range(X.shape[1]):
            K_j += kernel_func(X[:,i],X[:,j],param)
        K[:,j] -= K_j/X.shape[1]
    
    for i in range(X.shape[1]):
        K_t = 0
        for j in range(X.shape[1]):
            K_t += kernel_func(X[:,i],X[:,j],param)

    K[:,:] += K_t/(X.shape[1])**2

    return K

def topK_components(K_matrix, k):
    eigenvalues, eigenvectors, _ , _ = find_topK_eigenpairs(K_matrix,k)
    eigenvalues = np.sqrt(eigenvalues)
    alphas = (eigenvectors/(eigenvalues.reshape(1,k))[:,None]).reshape(1000,k)
    all_comp = []

    for i in range(K_matrix.shape[1]):
        data_i = []
        for k in range(alphas.shape[1]):
            comp_k = 0
            for j in range(K_matrix.shape[1]):

                comp_k += alphas[j,k]*K_matrix[i,j]

            data_i.append(comp_k)

        all_comp.append(data_i)

    return all_comp
            
def visualize_kernel_projection(dataset, X_uncentered, kernel_func, param):

    K = centered_K_matrix(X_uncentered,kernel_func,param)
    all_comp = topK_components(K,2)
    X = [i[0] for i in all_comp]
    Y = [i[1] for i in all_comp]

    labels = [data_point[1] for data_point in dataset]  

    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels)) 

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter([X[i] for i in indices], [Y[i] for i in indices], color=colors(label), label=f'{label}')

    plt.xlabel('1st component',size = 20)
    plt.ylabel('2nd component', size = 20)
    
    if kernel_func == polynomial_kernel:
        plt.title(f'Polynomial kernel d: {param}', size = 25)
    else:
        plt.title(f'RBF kernel with $\\sigma$: {param}', size = 25)
    
    plt.legend()
    plt.show()










