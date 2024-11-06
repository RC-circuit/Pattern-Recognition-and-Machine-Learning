import numpy as np
import matplotlib.pyplot as plt
import csv

def upload_csv(path):

    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    return np.array(data).astype(float)

def plot_datapoints(data):
    x = data[:,0]
    y = data[:,1]

    
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.title('Scatter Plot of the Datapoints')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xticks(np.linspace(min(x), max(x), num=10))
    plt.yticks(np.linspace(min(x), max(x), num=10))
    plt.show()