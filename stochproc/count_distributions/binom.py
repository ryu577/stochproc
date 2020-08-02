import numpy as np
from scipy.stats import binom

def expctd_cond_gr_m(m,n,p):
    if m<int(n*p):
        return sum(binom.pmf(np.arange(m+1,n+1),n,p)\
            /binom.sf(m,n,p)*np.arange(m+1,n+1))
    else:
        return n*p/binom.sf(m,n,p)-binom.cdf(m,n,p)\
                /binom.sf(m,n,p)*expctd_cond_leq_m(m,n,p)


def expctd_cond_leq_m(m,n,p):
    if m<=int(n/2):
        return sum(binom.pmf(np.arange(m+1),n,p)\
            /binom.cdf(m,n,p)*np.arange(m+1))
    else:
        return n*p/binom.cdf(m,n,p)-binom.sf(m,n,p)\
                /binom.cdf(m,n,p)*expctd_cond_gr_m(m,n,p)



