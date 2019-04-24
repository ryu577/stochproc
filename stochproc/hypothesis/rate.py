import numpy as np
from scipy.stats import poisson


def rate_hypothesis_test(lmb=12.0, mu=14.5):
    """
    Calculates the simulated p-value for the hypothesis test.
    args:
        lmb: The rate in interruptions per 100 VM years for first group.
        mu: The rate in interruptions per 100 VM years for second group.
    """
    # 10K VMs for 3 months; time in 100 VM years.
    t = 10e3/4/100

    # 15K VMs for 3 months; time in 100 VM years.
    s = 15e3/4/100

    pois1 = poisson.rvs(lmb*t,size=1)
    pois2 = poisson.rvs(mu*s,size=1)

    # Get the estimated rates.
    lmb_est = pois1/t
    mu_est = pois2/s

    lmb_mix = (pois1+pois2)/(s+t)

    d = mu_est-lmb_est
    p_val = 1-pois_diff_cdf(d[0],lmb_mix[0],t,s)
    return p_val


def pois_diff_cdf(d, lmb, t, s, nsim=100000):
    """
    Returns the probability that N_1/t-N_2/s<d
    where N_1~Pois(lmb*t) and N_2~Pois(lmb*s)
    """
    n1 = poisson.rvs(lmb*t,size=nsim)
    n2 = poisson.rvs(lmb*s,size=nsim)
    return sum(n1/t-n2/s < d)/nsim



