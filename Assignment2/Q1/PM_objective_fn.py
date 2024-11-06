import numpy as np
from Kmeans import objective_fn

def PM_objective_function(dataset,Lambdas, K):
    clusters = [[] for i in range(K)]
    for i in range (dataset.shape[0]):
        idx = np.argmax(Lambdas[i,:])
        clusters[idx].append(dataset[i,:])

    cluster_means = np.array([np.mean(clusters[i],axis=0) for i in range(K)])
    return objective_fn(clusters,cluster_means)


