import numpy as np
import matplotlib.pyplot as plt
import csv

def upload_csv(path):

    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return np.array(data).astype(float)