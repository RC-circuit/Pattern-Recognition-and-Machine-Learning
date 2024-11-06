import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial
from scipy.spatial import Voronoi, voronoi_plot_2d

def error_fn(clusters,cluster_means):
    error = 0
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            error += norm(clusters[i][j],cluster_means[i])

    return error
 
def calculate_mean(input):
    x_mean = 0
    y_mean = 0
    for i in range(len(input)):
        x_mean += input[i][0]
        y_mean += input[i][1]
    return [x_mean/len(input),y_mean/len(input)]


def norm(x1,x2):
    norm = 0
    for i in range(x1.shape[0]):
        norm += (x1[i] - x2[i])**2
    return norm
    

def K_means(data,k):
    error = []
    random_idx = random.sample(range(1,data.shape[0]),k)
    cluster_means = [list(data[i,:]) for i in random_idx]
    iter = 0
    new_cluster_means = [0 for i in range(k)]

    while(cluster_means != new_cluster_means):

        if iter != 0:
            cluster_means = new_cluster_means

        clusters = [[] for i in range(k)]
        
        for i in range(data.shape[0]):
            clusters[np.argmin([norm(data[i,:],mean) for mean in cluster_means])].append(data[i,:])

        new_cluster_means = [calculate_mean(clusters[i]) for i in range(k)]
        
        
        error.append(error_fn(clusters,new_cluster_means))

        iter += 1
        
    return clusters, new_cluster_means, error


def plot_clusters(clusters, cluster_means, error, k):
    plt.figure(figsize=(8, 6))
    colors = ['red','blue','green','orange','purple']
    for j in range(k):
        plt.scatter([clusters[j][i][0] for i in range(len(clusters[j]))], [clusters[j][i][1] for i in range(len(clusters[j]))], color=colors[j], label=f'Cluster {j+1}')
    plt.scatter([cluster_means[i][0] for i in range(k)],[cluster_means[i][1] for i in range(k)],c = 'black', marker = 'x', s = 100, label = 'Cluster means')
    try:
        plt.title(f'{k} Sets of Clusters and Error = {error[-1]:.2f}')
    except:
        plt.title(f'{k} Sets of Clusters and Error = {error:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(False)
    plt.show()


def Error_plot(error):
    plt.figure(figsize=(8, 6))
    plt.plot(error, marker='o', linestyle='-')
    plt.title('Error vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

def Voronoi_plot(cluster_means):
    vor = Voronoi(np.array(cluster_means),furthest_site=False)
    fig = voronoi_plot_2d(vor)
    plt.show()




