import numpy as np
import random as rand
from numpy.linalg import inv, det

def Gaussian_Log_Likelihood(mu,cov,pi,x):
    log_L = 0
    for i in range(x.shape[0]):
        sum = 0
        for k in range(pi.shape[0]):
            if det(cov[k,:,:]) != 0:
                exponent = -0.5*((x[i,:] - mu[k,:]).T) @ inv(cov[k,:,:]) @ (x[i,:] - mu[k,:])
                coefficient = pi[k]/(np.sqrt(2*((np.pi)**(x.shape[1]))*np.absolute(det(cov[k,:,:]))))

            else:
                cov[k,:,:] += (5e-2)*np.identity(x.shape[1])
                exponent = -0.5*((x[i,:] - mu[k,:]).T) @ inv(cov[k,:,:]) @ (x[i,:] - mu[k,:])
                coefficient = pi[k]/(np.sqrt(2*((np.pi)**(x.shape[1]))*np.absolute(det(cov[k,:,:]))))

            sum += np.exp(exponent)*coefficient    

        log_L += np.log(sum)

    return log_L


def Gaussian_EM_algorithm(mu_t, cov_t, pi_t, x, iterations):

    Log_Likelihood_list = []

    Lambda = np.random.rand(x.shape[0],pi_t.shape[0])
    Lambda = Lambda / Lambda.sum(axis=1)[:, np.newaxis]
    
    for iter in range(iterations):
        mu_t_1 = np.array([[0.0 for i in range(mu_t.shape[1])] for j in range(mu_t.shape[0])])
        cov_t_1 = np.zeros((pi_t.shape[0], mu_t.shape[1], mu_t.shape[1]))
        pi_t_1 = np.array([0.0 for i in range(pi_t.shape[0])])

        for i in range(x.shape[0]):
            for k in range(pi_t.shape[0]):
                if det(cov_t[k,:,:]) != 0:
                    exponent = -0.5*((x[i,:] - mu_t[k,:]).T) @ inv(cov_t[k,:,:]) @ (x[i,:] - mu_t[k,:])
                    coefficient = pi_t[k]/(np.sqrt(2*((np.pi)**(x.shape[1]))*np.absolute(det(cov_t[k,:,:]))))
                    prod = np.exp(exponent)*coefficient
                else:
                    cov_t[k,:,:] += (5e-2)*np.identity(x.shape[1],x.shape[1])
                    exponent = -0.5*((x[i,:] - mu_t[k,:]).T) @ inv(cov_t[k,:,:]) @ (x[i,:] - mu_t[k,:])
                    coefficient = pi_t[k]/(np.sqrt(2*((np.pi)**(x.shape[1]))*np.absolute(det(cov_t[k,:,:]))))
                    prod = np.exp(exponent)*coefficient
                
                Lambda[i,k] = prod
        
        Lambda = Lambda / Lambda.sum(axis=1)[:, np.newaxis]
        
        for k in range(pi_t.shape[0]):
            pi_t_1[k] = np.sum(Lambda[:,k])/x.shape[0]
            for i in range(x.shape[0]):
                mu_t_1[k,:] += Lambda[i,k]*x[i,:]/np.sum(Lambda[:,k])
                cov_t_1[k,:,:] += Lambda[i,k]*((x[i,:] - mu_t[k,:])@(x[i,:] - mu_t[k,:]).T)/np.sum(Lambda[:,k])
            
        mu_t = mu_t_1
        cov_t = cov_t_1
        pi_t = pi_t_1

        Log_Likelihood_list.append(Gaussian_Log_Likelihood(mu_t, cov_t, pi_t, x))
        
    return np.array(Log_Likelihood_list), Lambda
                