import numpy as np
from scipy.stats import binom_test, poisson, binom, nbinom, logser
from scipy.special import gamma

import matplotlib as mpl
import matplotlib.pyplot as plt

from stochproc.hypothesis.hypoth_tst_simulator import alpha_beta_tracer
from stochproc.count_distributions.compound_poisson import CompoundPoisson


## First the vanilla poisson.
lmb = 20
dist_rvs_poisson = lambda t: poisson.rvs(lmb*t)
lmb1=25
dist_rvs_poisson_1 = lambda t: poisson.rvs(lmb1*t)

alphas1,betas1,alpha_hats1 = alpha_beta_tracer(dist_rvs_poisson, dist_rvs_poisson_1,10,10)

## Now the mixed poisson.
def rvs_mxd_poisson(t, theta=5, m=100):
    p = theta/(theta+t)
    return nbinom.rvs(m,p)

rvs_mxd_poisson_1 = lambda t: rvs_mxd_poisson(t,4,100)

alphas2,betas2,alpha_hats2 = alpha_beta_tracer(rvs_mxd_poisson, rvs_mxd_poisson_1,10,10)

## And now the compound poisson.
def rvs_comp_poisson(t, theta=5, m=100):
    p = t/(theta+t)
    lamb = -m*np.log(1-p)
    log_rvs = lambda: logser.rvs(p)
    return CompoundPoisson.rvs_s_1(log_rvs,lamb)

rvs_comp_poisson_1 = lambda t: rvs_comp_poisson(t,theta=4,m=100)

alphas3,betas3,alpha_hats3 = alpha_beta_tracer(rvs_comp_poisson, rvs_comp_poisson_1,10,10)

## Now the plotting.
plt.plot(alphas1,betas1,label='UMP poisson on poisson')
plt.plot(alphas2,betas2,label='UMP poisson on mixed poisson')
plt.plot(alphas3,betas3,label='UMP poisson on compound poisson')
plt.legend()
plt.show()

