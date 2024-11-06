import numpy as np
import random as rand

def Bernoulli_Log_Likelihood(p,pi,x):
    log_L = 0
    for i in range(x.shape[0]):
        sum = 0
        for k in range(len(pi)):
            prod = pi[k]
            for m in range(len(p[0])):
                
                if x[i,m] == 1:
                    prod = prod*p[k][m]
                else:
                    prod = prod*(1-p[k][m])
            sum += prod

        log_L += np.log(sum)

    return log_L


def Bernoulli_EM_algorithm(p_t, pi_t, x, iterations):

    Log_Likelihood_list = []
    Lambda = np.random.rand(x.shape[0],len(pi_t))
    Lambda = Lambda / Lambda.sum(axis=1)[:, np.newaxis]
    
    for iter in range(iterations):
        p_t_1 = [[0 for i in range(len(p_t[0]))] for j in range(len(p_t))]
        pi_t_1 = [0 for i in range(len(pi_t))]

        for i in range(x.shape[0]):
            for k in range(len(pi_t)):
                prod = pi_t[k]
                for m in range(len(p_t[0])):
                    if x[i,m] == 1:
                        prod = prod*p_t[k][m]
                    else:
                        prod = prod*(1-p_t[k][m])
                Lambda[i,k] = prod
        
        Lambda = Lambda / Lambda.sum(axis=1)[:, np.newaxis]

        
        for k in range(len(pi_t)):
            pi_t_1[k] = np.sum(Lambda[:,k])/x.shape[0]
            for m in range(len(p_t[0])):
                p_t_1[k][m] = np.sum(Lambda[:,k]*x[:,m])/np.sum(Lambda[:,k])

        p_t = p_t_1
        pi_t = pi_t_1

        Log_Likelihood_list.append(Bernoulli_Log_Likelihood(p_t,pi_t,x))

    return np.array(Log_Likelihood_list), Lambda
                

    
