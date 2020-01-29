import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

os.chdir("/Users/isaac.kleisle-murphy/Documents/Swat_2020/Data_Privacy/Bayesian-Inference-with-Python/MATH347RLabs")
CEsample = pd.read_csv('CEsample1.csv')



#CEsample['LogTotalExpLastQ'] = np.log(data.loc[:, 'TotalExpLastQ'])
CEsample['LogTotalExpLastQ'] = np.log(CEsample.loc[:, 'TotalExpLastQ'])
CEsample.head()



def gibbs_normal(input, S, seed): 
    np.random.seed(seed)
    ybar = np.mean(input['y'])
    n = len(input['y'])
    para = np.zeros(shape=(S,2))
    phi = input['phi_init']

    for s in range(0, S): ##Needs to be S, not S-1; otherwise the last 0 from np.zeros is in our sample
        
        mu1 = (input['mu_0']/(input['sigma_0'] ** 2) + n*phi*ybar)/ \
            (1/(input['sigma_0'] ** 2) + n * phi)
        sigma1 = np.sqrt(1/(1/(input['sigma_0'] ** 2) + n*phi))
        mu = np.random.normal(mu1, sigma1, 1)
        alpha1 = input['alpha'] + n/2
        beta1 = input['beta'] + np.sum((input['y'] - mu) ** 2)/2
        phi = np.random.gamma(alpha1, 1/beta1, 1) #note that np.random.gamma takes scale, not rate
        para[s] = [mu, phi]

    return para


y = CEsample.LogTotalExpLastQ
mu_0 = 8
sigma_0 = 1
alpha = 1
beta = 1
phi_init = 1

input = {'y': y, 'mu_0': mu_0, 'sigma_0': sigma_0, 'alpha': alpha, \
       'beta': beta, 'phi_init': phi_init}

output = gibbs_normal(input, S = 5000, seed = 123)


sns.distplot(output[:, 0], hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'mu')
