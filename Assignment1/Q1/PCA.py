import numpy as np
import matplotlib.pyplot as plt
import operator

def covariance_matrix(X):

    return (1/X.shape[1])*(X @ (X.T))


def find_topK_eigenpairs(input_matrix, K):

    
    eigenvalues , eigenvectors = np.linalg.eig(input_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    variance_percent = [eigenvalues[i]/np.sum(eigenvalues)*100 for i in range(len(eigenvalues))]
    
    idx = eigenvalues.argsort()[::-1] 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    info_captured_percent = np.sum(eigenvalues[0:K])/np.sum(eigenvalues)
    return eigenvalues[0:K], eigenvectors[:,0:K], variance_percent[0:K], info_captured_percent

def EV_ninetyfive_percent(input_matrix):
    for K in range(input_matrix.shape[0]):
        _,_,_, percent = find_topK_eigenpairs(input_matrix,K)
        if percent >= 0.95:
            break
    
    return K


def reconstruct(X, EigenVectors, mean):
    Xnew = np.zeros([X.shape[0],X.shape[1]])

    for i in range(X.shape[1]):
        for j in range(EigenVectors.shape[1]):
            
            Xnew[:,i] += ((X[:,i].T) @ EigenVectors[:,j])*EigenVectors[:,j]
    
    return Xnew + mean


def visualize_eigenvectors(variance_percent,eigenvectors,num_samples,cols, figsize = (15,15)):

    plt.figure(figsize= figsize)

    for i in range(num_samples):
        plt.subplot(int(num_samples/cols) +1, cols, i+1)
        plt.imshow(eigenvectors[:,i].reshape(28,28))
        plt.axis('off')
        plt.rc('figure', titlesize = 16)
        plt.title(f'PC: {i+1} ; variance: {variance_percent[i]:.2f} %', size = 16)

    plt.suptitle('Principle Components')
    plt.tight_layout(h_pad=2)
    plt.subplots_adjust(top=0.85)
    plt.show()

def reconstruct_plots(X, mean, input_matrix, d):
    fig = plt.figure(figsize=(15,15), dpi =80)
    for i,ele in enumerate(d):

        _, eigenvectors, _, _ = find_topK_eigenpairs(input_matrix,ele)
        Xnew = reconstruct(X, eigenvectors,mean)

        plt.subplot(len(d),2,2*i+1)
        plt.imshow((X[:,0] + mean[:,0]).reshape(28,28))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(len(d),2,2*i+2)
        plt.imshow((Xnew[:,0]).reshape(28,28))
        plt.title(f'Reconstructed Image d:{ele}')
        plt.axis('off')

    plt.suptitle('Reconstruction')
    plt.tight_layout(h_pad=2)
    plt.show()