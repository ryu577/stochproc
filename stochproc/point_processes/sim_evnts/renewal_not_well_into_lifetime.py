import numpy as np
from scipy.stats import lomax
from scipy.special import gamma


def sim_weibull(intr_strt=20, win_size=10, k=5.5, lmb=1):
    """
    [1] https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html
    [2] https://en.wikipedia.org/wiki/Weibull_distribution#:~:text=The%20Weibull%20distribution%20is%20the%20maximum%20entropy%20distribution%20for%20a,ln(%CE%BBk)%20%E2%88%92%20.
    """
    mean = lmb*gamma(1+1/k)
    print(1/mean)
    catches = 0
    for _ in range(10000):
        j = intr_strt
        t_i = 0
        while t_i < j+50:
            t_i += lmb*(-np.log(np.random.uniform()))**(1/k)
            if j < t_i and t_i < j+win_size:
                catches += 1
    print(catches/10000/win_size)
    # 1.1241876
    # 1.1184, 1.1259


def sim_lomax(intr_strt=20):
    c = 1.88
    mean, var, skew, kurt = lomax.stats(c, moments='mvsk')
    print(1/mean)
    catches = 0
    for _ in range(10000):
        j = intr_strt
        t_i = 0
        while t_i < j+50:
            t_i += lomax.rvs(c)
            if j < t_i and t_i < j+1:
                catches += 1
    print(catches/10000)
    # 0.8799999999999998
    # 0.8873, 8677
