from datasets import load_dataset
import numpy as np
from dataset import custom_dataset, show_images, flattened_centered_data_matrix
from PCA import find_topK_eigenpairs, reconstruct, visualize_eigenvectors, covariance_matrix, EV_ninetyfive_percent, reconstruct_plots
from kernel_PCA import centered_K_matrix, topK_components, polynomial_kernel, radial_basis_kernel, visualize_kernel_projection
import matplotlib.pyplot as plt

data = load_dataset("mnist")

dataset = custom_dataset(data,1000,10)

show_images(dataset,12,4)

mean, X = flattened_centered_data_matrix(dataset)
cov_matrix = covariance_matrix(X)
K = 130
eigenvalues, eigenvectors, variance_percent, info_capture_percent = find_topK_eigenpairs(cov_matrix,K)

visualize_eigenvectors(variance_percent, eigenvectors,12,4,(15,15))
print(info_capture_percent)

print(EV_ninetyfive_percent(cov_matrix))
num_samples = 12
cols = 4
Xnew = reconstruct(X, eigenvectors,mean)
fig = plt.figure(figsize=(15,15), dpi =80)
for i in range(num_samples):

    plt.subplot(int(num_samples/cols)+1,cols,i+1)
    plt.imshow((Xnew[:,i]).reshape(28,28))
    plt.axis('off')

plt.suptitle(f'Reconstruction with d:{K}', size = 40)
plt.tight_layout(h_pad=2)
plt.subplots_adjust(top=0.85)
plt.show()


d = [784, 600, 500, 300, 130, 50]
reconstruct_plots(X,mean,cov_matrix,d)


visualize_kernel_projection(dataset, X+mean, radial_basis_kernel, 10000)

