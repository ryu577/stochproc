from scipy.stats import binom_test, poisson, binom
import numpy as np


def binom_tst_beta(p_null=0.5,p_alt=0.6,n=10,alpha_hat=0.05):
    x_a = binom.isf(alpha_hat,n,p_null)
    return binom.cdf(x_a,n,p_alt)


def binom_tst_beta_sim(p_null=0.5,p_alt=0.6,n=10,alpha_hat=0.05,n_sim=1000):
    #Generate from the alternate.
    rvs = binom.rvs(n,p_alt,size=n_sim)
    #Check against the null.
    p_vals = np.array([binom_test(i,n,p_null,alternative='greater') \
                for i in rvs])
    return sum(p_vals>alpha_hat)/len(rvs)

