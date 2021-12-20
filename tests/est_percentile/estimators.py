import numpy as np
from scipy.stats import norm, lognorm, expon, lomax, weibull_min
import matplotlib.pyplot as plt


def prcntl(a, q, interpolate=1):
    a = sorted(a)
    n = len(a)
    lt = int(np.floor(q*(n-1)))
    frac = 0
    if interpolate==1:
        frac = np.modf(q*(n-1))[0]
    elif interpolate==2:
        frac = expon_frac(q,n)
    return a[lt]*(1-frac)+a[lt+1]*frac


def rvs_fn1(n):
    return norm.rvs(10, 1, size=n)

def rvs_fn2(n):
    return lognorm.rvs(1, 0, size=n)

def rvs_fn3(n):
    return np.random.exponential(size=n)

def rvs_fn4(n):
    return lomax.rvs(c=3,size=n)

def rvs_fn5(n):
    return weibull_min.rvs(c=.5,size=n)

def ppf_fn1(q):
    return norm.ppf(q, 10, 1)

def ppf_fn2(q):
    return lognorm.ppf(q, 1, 0)

def ppf_fn3(q):
    return expon.ppf(q)

def ppf_fn4(q):
    return lomax.ppf(q,c=4)

def ppf_fn5(q):
    return weibull_min.ppf(q,c=.5)

def expon_frac(q, n):
    """
    TODO: Move this inside the library.
    """
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)
    return (-np.log(1-q)-summ)*(n-lt-1)


def coeff_variation_and_bias(n=100, rvs_fn=rvs_fn2, ppf_fn=ppf_fn2):
    u_errs = []
    u_errs1 = []
    u_stds = []
    u_stds1 = []
    u_coff_vars = []
    u_medians = []
    u_medians1 = []

    for q in np.arange(0.05, 1, 0.05):
        errs = []
        errs1 = []
        ests = []
        ests1 = []
        for _ in range(10000):
            x = rvs_fn(n)
            real_val = ppf_fn(q)
            est_val = np.percentile(x, q*100)
            est_val1 = prcntl(x, q, 2)
            err = (real_val-est_val)
            err1 = (real_val-est_val1)
            errs.append(err)
            errs1.append(err1)
            ests.append(est_val)
            ests1.append(est_val1)

        print(np.mean(errs))
        u_errs.append(np.mean(errs))
        u_errs1.append(np.mean(errs1))
        u_stds.append(np.std(ests))
        u_stds1.append(np.std(ests1))
        u_medians.append(np.median(errs))
        u_medians1.append(np.median(errs1))

        u_coff_vars.append(np.std(ests)/np.mean(ests))
        #plt.hist(errs)
        #plt.show()

    qs = np.arange(0.05, 1, 0.05)
    plt.plot(qs, u_errs, label="Average bias")
    plt.plot(qs, u_errs1, label="Average bias 1")
    plt.plot(qs, u_stds, label="standard deviation")
    plt.plot(qs, u_stds1, label="standard deviation 1")
    plt.plot(qs, u_medians, label="Median bias")
    plt.plot(qs, u_medians1, label="Median bias 1")
    plt.axhline(0, color="black")
    plt.axvline(0.5, color="black")
    plt.legend()
    plt.xlabel("Percentile")
    plt.show()


##################
import os

basedir = '.\\plots\\sample_\\'
if os.name == 'posix':
    basedir = 'plots/sample_'


for sampl in np.arange(35, 220, 5):
    q = 0.9
    errs = []
    ests = []
    real_val = ppf_fn2(q)
    for _ in range(100000):
        x = rvs_fn2(sampl)
        est_val = np.percentile(x, q*100)
        err = (real_val-est_val)/real_val
        errs.append(err)
        ests.append(est_val)

    print(np.mean(errs))

    plt.hist(ests, bins=np.arange(0,14,1))
    plt.axvline(real_val, label="actual", color="yellow")
    plt.axvline(np.mean(ests), label="average", color="green")
    plt.axvline(np.percentile(ests, 100-q*100), label="percentile of estimates", color="orange")
    plt.axvline(np.percentile(ests, .5*100), label="median of estimates", color="purple")
    plt.legend()
    plt.savefig('plots/sample_' + str(sampl) + '.png')
    plt.close()
    print('processed sample size ' + str(sampl))

