import numpy as np
from scipy.stats import weibull_min
from scipy.stats import lomax


def sim_poisson_simplified(vms=1000, s=500, e=530, lmb=15):
    catches = 0
    durtn = (e-s)
    for _ in range(vms):
        t_i = 0
        while t_i < e+500:
            t_i += np.random.exponential(lmb)
            if s < t_i and t_i < e:
                catches += 1
    return catches/(vms*durtn)


def sim_bimodal():
    catches = 0
    for _ in range(50000):
        j = np.random.uniform()*1000#+ np.random.normal(3,10)
        #j = np.random.exponential(500)
        t_i = 0
        while t_i < j+100:
            if np.random.uniform() < 0.5:
                t_i += 10
            else:
                t_i += 20
            if j < t_i and t_i < j+1:
                catches += 1
    print(catches/50000)


def sim_poisson():
    catches = 0
    for _ in range(10000):
        j = np.random.uniform()*1000 + np.random.normal(0,10)
        t_i = 0
        while t_i < j+500:
            t_i += np.random.exponential(15)
            if j < t_i and t_i < j+1:
                catches += 1
    print(catches/10000)


def sim_poisson_v2():
    catches = 0
    catches2 = 0
    total_t = 0
    for _ in range(10000):
        j = np.random.uniform()*1000
        t_i = 0
        tt = 0
        catches1 = -1
        while t_i < j+500:
            t_i += np.random.exponential(15)
            if j < t_i and t_i < j+30:
                tt = t_i
                catches += 1
                catches1 += 1
            total_t += max((tt-j), 0)
            catches2 += max(0, catches1)
    print(catches/10000/30)
    print(catches2/total_t)


def sim_weibull_min():
    c = 1.79
    mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
    print(1/mean)
    catches = 0
    for _ in range(10000):
        j = np.random.uniform()*1000
        t_i = 0
        while t_i < j+500:
            t_i += weibull_min.rvs(c)
            if j < t_i and t_i < j+1:
                catches += 1
    print(catches/10000)
    # 1.1241876
    # 1.1184, 1.1259


def sim_weibull_min_v2():
    c = 1.79
    mean, var, skew, kurt = weibull_min.stats(c, moments='mvsk')
    catches = 0
    catches2 = 0
    total_t = 0
    for _ in range(20000):
        j = np.random.uniform()*50
        t_i = 0
        tt = 0
        catches1 = -1
        while t_i < j+100:
            t_i += weibull_min.rvs(c)
            if j < t_i and t_i < j+30:
                tt = t_i
                catches += 1
                catches1 += 1
            total_t += max((tt-j), 0)
            catches2 += max(0, catches1)
    print(catches/20000/30)
    print(catches2/total_t)


def sim_lomax():
    c = 1.88
    mean, var, skew, kurt = lomax.stats(c, moments='mvsk')
    print(1/mean)
    catches = 0
    for _ in range(10000):
        j = np.random.uniform()*1000
        t_i = 0
        while t_i < j+500:
            t_i += lomax.rvs(c)
            if j < t_i and t_i < j+1:
                catches += 1
    print(catches/10000)
    # 0.8799999999999998
    # 0.8873, 8677


def sim_lomax_v2():
    c = 1.88
    catches = 0
    catches2 = 0
    total_t = 0
    for _ in range(20000):
        j = np.random.uniform()*1000
        t_i = 0
        tt = 0
        catches1 = -1
        while t_i < j+100:
            t_i += lomax.rvs(c)
            if j < t_i and t_i < j+30:
                tt = t_i
                catches += 1
                catches1 += 1
            total_t += max((tt-j), 0)
            catches2 += max(0, catches1)
    print(catches/20000/30)
    print(catches2/total_t)
