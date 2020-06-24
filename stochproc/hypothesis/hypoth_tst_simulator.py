import numpy as np
from scipy.stats import binom_test, poisson, binom, nbinom
from scipy.special import gamma
from stochproc.count_distributions.compound_poisson import CompoundPoisson
#from stochproc.count_distributions.interarrival_weibull import InterarrivalWeibull
from stochproc.hypothesis.rate_test import rateratio_test, rateratio_test_two_sided
from datetime import datetime


def alpha_beta_curve(rvs_fn, n_sim=10000, lmb=20, t1=10, t2=3, \
                        scale=1.0, dellmb = 10.0, hypoth_fn=rateratio_test):
    ## First the null hypothesis..
    alpha_hats = np.concatenate((np.arange(0.000000000001,0.0099,0.0000001),
                                        np.arange(0.01,1.00,0.001), 
                                        np.arange(0.991,1.00,0.001)),axis=0)
    alphas = np.zeros(len(alpha_hats))

    ## First generate from null and find alpha_hat and alpha.
    for _ in range(n_sim):
        m1 = rvs_fn(lmb,t1)
        m2 = rvs_fn(lmb,t2)
        p_val = hypoth_fn(m1,t1,m2,t2,scale)
        alphas += (p_val < alpha_hats)/n_sim

    ## Now the alternate hypothesis
    betas = np.zeros(len(alpha_hats))
    for _ in range(n_sim):
        m1 = rvs_fn(lmb,t1)
        m2 = rvs_fn((lmb+dellmb),t2)
        p_val = hypoth_fn(m1,t1,m2,t2,scale)
        betas += 1/n_sim - (p_val < alpha_hats)/n_sim
    return alphas, betas, alpha_hats


def alpha_beta_tracer(rvs_fn_1, rvs_fn_2, t1=10, t2=10, n_sim=10000, scale=1.0, hypoth_tst=rateratio_test):
    ## First the null hypothesis..
    alpha_hats = np.concatenate((np.arange(0.000000000001,0.0099,0.0000001),
                                        np.arange(0.01,1.00,0.001), 
                                        np.arange(0.991,1.00,0.001)),axis=0)
    alphas = np.zeros(len(alpha_hats))
    ## First generate from null and find alpha_hat and alpha.
    for _ in range(n_sim):
        m1 = rvs_fn_1(t1)
        m2 = rvs_fn_1(t2)
        p_val = hypoth_tst(m1,t1,m2,t2,scale)
        alphas += (p_val < alpha_hats)/n_sim

    ## Now the alternate hypothesis
    betas = np.zeros(len(alpha_hats))
    for _ in range(n_sim):
        #TODO: recycle the m1 or m2 from alpha simulation.
        m1 = rvs_fn_1(t1)
        m2 = rvs_fn_2(t2)
        p_val = hypoth_tst(m1,t1,m2,t2,scale)
        betas += 1/n_sim - (p_val < alpha_hats)/n_sim
    return alphas, betas, alpha_hats


# def dist_rvs_interarrivalw(lmb_target=20,t=20):
#     k=0.5; lmb_target = 20
#     ## This is approximate. 
#     w_lmb = 1/lmb_target/gamma(1+1/k)
#     iw = InterarrivalWeibull(k,w_lmb,t)
#     return iw.rvs1()


def run_simulns(fn, hypoth_fn=rateratio_test, \
                n_sim=50000, lmb=20.0, t1=10.0, t2=3.0, scale=1.0):
    time1=datetime.now()
    alphas1,betas1,alpha_hats1 = alpha_beta_curve(fn, 
                                    n_sim=n_sim, 
                                    lmb=lmb,
                                    t1=t1, t2=t2, 
                                    scale=scale, hypoth_fn=hypoth_fn)
    time2=datetime.now()
    time_del = (time2-time1).seconds
    print("Time taken in seconds: " + str(time_del))
    return alphas1, betas1, alpha_hats1

