import numpy as np
import random
import matplotlib.pyplot as plt

def objective_fn(clusters,cluster_means):
    error = 0
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            error += np.linalg.norm(np.array(clusters[i][j]) - np.array(cluster_means[i]))

    return error
    
def K_means(data,k):
    error = []
    random_idx = random.sample(range(1,data.shape[0]),k)
    cluster_means = np.array([data[i,:] for i in random_idx])
    iter = 0
    new_cluster_means = np.array([[0 for j in range(data.shape[1])] for i in range(k)])

    while((cluster_means-new_cluster_means).any()):

        if iter != 0:
            cluster_means = new_cluster_means

        clusters = [[] for i in range(k)]
        
        for i in range(data.shape[0]):
            
            clusters[np.argmin([np.linalg.norm(data[i,:] - mean) for mean in cluster_means])].append(data[i,:])

        new_cluster_means = np.array([np.mean(clusters[i],axis=0) for i in range(k)])
        
        
        error.append(objective_fn(clusters,new_cluster_means))

        iter += 1
        
    return clusters, new_cluster_means, error

def Error_plot(error):
    plt.figure(figsize=(8, 6))
    plt.plot(error, marker='o', linestyle='-')
    plt.title('Objective function vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Objective function')
    plt.show()