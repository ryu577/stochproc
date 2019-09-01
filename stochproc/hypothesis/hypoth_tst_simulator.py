import numpy as np
from scipy.stats import binom_test, poisson
from stochproc.count_distributions.compound_poisson import CompoundPoisson
from stochproc.count_distributions.interarrival_weibull import InterarrivalWeibull
import matplotlib.pyplot as plt


def rateratio_test(n1,t1,n2,t2,scale=1.0):
    n2, n1 = n2/scale, n1/scale
    p_val = binom_test(n2,n1+n2,t2/(t1+t2),alternative='greater')
    return p_val


def alpha_beta_curve(n_sim=10000, lmb=20, n=32, p=0.7, t1=10, t2=3, 
                    distr='poisson', scale=1.0):
    ## First the null hypothesis..
    alpha_hats = np.concatenate((np.arange(0.000001,0.0099,0.0001), 
                                np.arange(0.01,1.00,0.01), 
                                np.arange(0.991,1.00,0.001)),axis=0)
    alphas = np.zeros(len(alpha_hats))

    ## First generate from null and find alpha_hat and alpha.
    for _ in range(n_sim):
        if distr == 'poisson':
            m1 = poisson.rvs(lmb*t1)
            m2 = poisson.rvs(lmb*t2)
        else:
            m1 = CompoundPoisson.rvs_s(lmb*t1,n,p)
            m2 = CompoundPoisson.rvs_s(lmb*t2,n,p)
        p_val = rateratio_test(m1,t1,m2,t2,scale)
        alphas += (p_val < alpha_hats)/n_sim

    ## Now the alternate hypothesis
    dellmb = 10.0
    betas = np.zeros(len(alpha_hats))
    for _ in range(n_sim):
        if distr == 'poisson':
            m1 = poisson.rvs(lmb*t1)
            m2 = poisson.rvs((lmb+dellmb)*t2)
        else:
            m1 = CompoundPoisson.rvs_s(lmb*t1,n,p)
            m2 = CompoundPoisson.rvs_s((lmb+dellmb)*t2,n,p)        
        p_val = rateratio_test(m1,t1,m2,t2,scale)
        betas += 1/n_sim - (p_val < alpha_hats)/n_sim
    return alphas, betas, alpha_hats


n=32; p=0.7
dist_rvs = lambda lmb: CompoundPoisson.rvs_s(lmb,n,p)

alphas1,betas1,alpha_hats1 = alpha_beta_curve(n_sim=50000)
plt.plot(alphas1,betas1,label='UMP poisson on poisson')

alphas2,betas2,alpha_hats2 = alpha_beta_curve(n_sim=50000,distr='compound_poisson')
plt.plot(alphas2,betas2,label='UMP poisson on compound poisson')

"""
alphas,betas = alpha_beta_curve(distr='compound_poisson',scale=30.0)
plt.plot(alphas,betas,label='scaled down poisson on compound poisson')

alphas,betas = alpha_beta_curve(distr='compound_poisson',scale=1/30.0)
plt.plot(alphas,betas,label='scaled up poisson on compound poisson')
"""

plt.legend()
plt.show()

