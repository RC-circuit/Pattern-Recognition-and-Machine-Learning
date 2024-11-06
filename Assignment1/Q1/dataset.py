
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def custom_dataset(data, data_size, num_classes):

    label_count = {}
    for label in range(0, num_classes):
        label_count["count{0}".format(label)] = 0
    
    dataset = []

    for i in range(0,len(data['train'])):
        key, count = list(label_count.items())[data['train'][i]['label']]

        if count < int(data_size/num_classes):
            dataset.append((np.array(data['train'][i]['image']),data['train'][i]['label']))
            label_count[key] += 1

        if len(dataset) == data_size:
            break

    return dataset


def show_images(dataset, num_samples, cols):

    plt.figure(figsize=(10,10))

    for i, ele in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols)+1, cols, i + 1)
        plt.imshow(ele[0])
        plt.title(ele[1])
        plt.axis('off')
        
    plt.suptitle('MNIST Dataset')
    plt.tight_layout()
    plt.show()


def flattened_centered_data_matrix(dataset):
    X = np.empty([784,1000])
    for i, ele in enumerate(dataset):
        X[:,i] = (ele[0].reshape(784))

    mean = np.array([np.mean(X,axis=1),]*1000).T
    return mean, X - mean