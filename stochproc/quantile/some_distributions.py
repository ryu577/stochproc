from scipy.stats import norm, lognorm, expon, lomax, weibull_min
import numpy as np


def rvs_fn1(n):
    return norm.rvs(10, 1, size=n)


def rvs_fn2(n):
    return lognorm.rvs(1, 0, size=n)


def rvs_fn3(n):
    return np.random.exponential(size=n)


def rvs_fn4(n):
    return lomax.rvs(c=.9, size=n)


def rvs_fn5(n):
    return weibull_min.rvs(c=5, size=n)


def ppf_fn1(q):
    return norm.ppf(q, 10, 1)


def ppf_fn2(q):
    return lognorm.ppf(q, 1, 0)


def ppf_fn3(q):
    return expon.ppf(q)


def ppf_fn4(q):
    return lomax.ppf(q, c=.9)


def ppf_fn5(q):
    return weibull_min.ppf(q, c=5)
