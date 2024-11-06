import numpy as np
import random as rand
import matplotlib.pyplot as plt
import csv
from initialization import bernoulli_initializer, gaussian_initializer
from Bernoulli_approach import Bernoulli_EM_algorithm
from Gaussian_approach import Gaussian_EM_algorithm
from Kmeans import K_means, Error_plot
from PM_objective_fn import PM_objective_function

def upload_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        
    return np.array(data).astype(float)


dataset = upload_csv('/home/ruthwik/Sem 6/PRML/assignment2/A2Q1.csv')
iterations = 20
avg_log_likelihood = []
for i in range(100):
    p, pi = bernoulli_initializer(dataset,4)
    log_likelihood, Lambda = Bernoulli_EM_algorithm(p, pi, dataset, iterations)
    avg_log_likelihood.append(log_likelihood)

plt.plot(range(0,iterations),np.average(np.array(avg_log_likelihood), axis = 0))
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Bernoulli Mixture')
plt.show()
    
iterations = 15
avg_log_likelihood = []
for i in range(100):
    mu, cov, pi = gaussian_initializer(dataset,4)
    log_likelihood, Lambda = Gaussian_EM_algorithm(mu, cov, pi, dataset, iterations)
    avg_log_likelihood.append(log_likelihood)


plt.plot(range(0,iterations),np.average(np.array(avg_log_likelihood), axis = 0))
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')
plt.title('Gaussian Mixture')
plt.show()


_, _, error = K_means(dataset,4)
print(error[-1])
Error_plot(error)


p, pi = bernoulli_initializer(dataset,4)
log_likelihood, Lambda = Bernoulli_EM_algorithm(p, pi, dataset, 20)

print(PM_objective_function(dataset,Lambda, 4))

mu, cov, pi = gaussian_initializer(dataset,4)
log_likelihood, Lambda = Gaussian_EM_algorithm(mu, cov, pi, dataset, 15)
print(PM_objective_function(dataset,Lambda, 4))