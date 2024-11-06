from dataset import upload_csv, plot_datapoints
from Kmeans import K_means, plot_clusters, Error_plot, Voronoi_plot
from spectral_clustering import spectral_clustering, K_matrix, polynomial_kernel, radial_basis_kernel, alternate_clustering
import matplotlib.pyplot as plt
import numpy as np

data = upload_csv('/home/ruthwik/Sem 6/PRML/dataset.csv')
plot_datapoints(data)
K = 5
clusters, cluster_means, error = K_means(data,K)
plot_clusters(clusters,cluster_means, error, K)
Error_plot(error)
Voronoi_plot(cluster_means)

spectral_clustering(K_matrix, data, polynomial_kernel, 3, 2)
alternate_clustering(K_matrix, data, radial_basis_kernel, 100, 2)



