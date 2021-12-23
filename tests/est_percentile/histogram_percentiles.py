##################
import os
import numpy as np
from scipy.stats import lognorm, lomax
import matplotlib.pyplot as plt


basedir = '.\\plots\\sample_\\'
if os.name == 'posix':
    basedir = 'plots/sample_'
plt.style.use('dark_background')

def ppf_fn2(q):
    return lognorm.ppf(q, 1, 0)

def ppf_fn4(q):
    return lomax.ppf(q,c=.9)

def rvs_fn2(n):
    return lognorm.rvs(1, 0, size=n)

def rvs_fn4(n):
    return lomax.rvs(c=.9,size=n)

for sampl in np.arange(5, 55, 3):
    q = 0.5
    errs = []
    ests = []
    real_val = ppf_fn4(q)
    for _ in range(100000):
        x = rvs_fn4(sampl)
        est_val = np.percentile(x, q*100)
        err = (real_val-est_val)/real_val
        errs.append(err)
        ests.append(est_val)

    print(np.mean(errs))

    plt.hist(ests, bins=np.arange(0,10,.25),color="orange")
    plt.title("Distribution of empirical medians for the Lomax \ndistribution from sample size: " + str(sampl))
    plt.axvline(real_val, label="Actual median", color="yellow")
    plt.axvline(np.mean(ests), label="Average of estimates", color="green")
    #plt.axvline(np.percentile(ests, 100-q*100), label="percentile of estimates", color="orange")
    plt.axvline(np.percentile(ests, .5*100), label="Median of estimates", color="purple")
    plt.legend()
    plt.savefig('plots/sample_' + str(sampl) + '.png')
    plt.close()
    print('processed sample size ' + str(sampl))

