from Kmeans import K_means, plot_clusters, Error_plot, calculate_mean, error_fn
import numpy as np
import matplotlib.pyplot as plt
import math

def polynomial_kernel(x1, x2, d):
    return ((x1.T)@(x2) + 1)**d

def radial_basis_kernel(x1, x2, sigma):
    return math.exp( (-1*((x1 - x2).T)@(x1 - x2))/ (2*(sigma**2)) )


def find_topK_eigenpairs(input_matrix, K):
    eigenvalues , eigenvectors = np.linalg.eig(input_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    idx = eigenvalues.argsort()[::-1] 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    info_captured_percent = np.sum(eigenvalues[0:K])/np.sum(eigenvalues)
    return eigenvalues[0:K], eigenvectors[:,0:K], info_captured_percent


def K_matrix(X, kernel_func, param):
    K = np.empty([X.shape[0], X.shape[0]])
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i,j] = kernel_func(X[i,:],X[j,:],param)

    return K

def plot_kernel_clusters(clusters, cluster_means, error, kernel_func, params):
    plt.figure(figsize=(8, 6))
    colors = ['red','blue']
    for j in range(2):
        plt.scatter([clusters[j][i][0] for i in range(len(clusters[j]))], [clusters[j][i][1] for i in range(len(clusters[j]))], color=colors[j], label=f'Cluster {j+1}')
    plt.scatter([cluster_means[i][0] for i in range(2)],[cluster_means[i][1] for i in range(2)],c = 'black', marker = 'x', s = 100, label = 'Cluster means')
    try:
        plt.title(f'Clusters using {kernel_func} {params} and Error = {error[-1]:.2f}')
    except:
        plt.title(f'Clusters using {kernel_func} {params} and Error = {error:.2f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def spectral_clustering(K_matrix, X, kernel_func, param, k):

    _, topK_eigenvectors, _ = find_topK_eigenpairs(K_matrix(X,kernel_func, param),k)
    normalized_EVmatrix = np.empty_like(topK_eigenvectors)
    for i in range(X.shape[0]):
        for j in range(k):
            normalized_EVmatrix[i,j] = topK_eigenvectors[i,j]/np.sqrt(np.sum(np.square(topK_eigenvectors[i,:])))

    clusters, cluster_means, error = K_means(normalized_EVmatrix, k)
    if kernel_func == polynomial_kernel:
        plot_kernel_clusters(clusters, cluster_means, error, 'Polynomial kernel d:', param)
    else:
        plot_kernel_clusters(clusters, cluster_means, error, 'RBF kernel sigma:', param)
    return None
    

def alternate_clustering(K_matrix, X, kernel_func, param, k):
    _, topK_eigenvectors, _ = find_topK_eigenpairs(K_matrix(X,kernel_func, param),k)
    normalized_EVmatrix = np.empty_like(topK_eigenvectors)
    for i in range(X.shape[0]):
        for j in range(k):
            normalized_EVmatrix[i,j] = topK_eigenvectors[i,j]/np.sqrt(np.sum(np.square(topK_eigenvectors[i,:])))

    clusters = [[] for i in range(k)]
        
    for i in range(X.shape[0]):
        clusters[np.argmax(normalized_EVmatrix[i,:])].append(normalized_EVmatrix[i,:])

    cluster_means = [calculate_mean(clusters[i]) for i in range(k)]
    error = error_fn(clusters, cluster_means)
    if kernel_func == polynomial_kernel:
        plot_kernel_clusters(clusters, cluster_means, error, 'Polynomial kernel d:', param)
    else:
        plot_kernel_clusters(clusters, cluster_means, error, 'RBF kernel sigma:', param)
    return None
