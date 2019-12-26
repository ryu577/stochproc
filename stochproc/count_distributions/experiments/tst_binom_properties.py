import numpy as np

## For small values of alpha, the first term is smaller.
# For larger values, the second term is smaller. This affects
# alpha_hat to alpha profile.
def tst(alp,l,k,p):
    return binom.isf(alp,l*k,p),l*binom.isf(alp,k,p)


