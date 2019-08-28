import numpy as np
from scipy.special import comb

def p_n1(t1, t2, n1, n2):
    n=n1+n2; t=t1+t2
    return t1**n1*t2**n2/(t**n*comb(n,n1))


