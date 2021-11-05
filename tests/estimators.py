import numpy as np
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt


def estimate_percentile(x, q, is_sorted=False):
    n = len(x)
    if not is_sorted:
        x = sorted(x)
    lo = int(np.floor(n*q))
    hi = int(np.ceil(n*q))
    x1 = x[lo]
    x2 = x[hi]
    d = q*n-lo
    return x1*(1-d)+x2*d


def rvs_fn1(n):
    return norm.rvs(0, 1, size=n)


def rvs_fn2(n):
    return lognorm.rvs(1, 0, size=n)


def ppf_fn1(q):
    return norm.ppf(q, 0, 1)


def ppf_fn2(q):
    return lognorm.ppf(q, 1, 0)


def coeff_variation_and_bias():
    u_errs = []
    u_stds = []
    u_coff_vars = []

    for q in np.arange(0.1, 1, 0.1):
        #q = 0.9
        errs = []
        ests = []
        for _ in range(10000):
            x = rvs_fn2(1000)
            real_val = ppf_fn2(q)
            est_val = estimate_percentile(x, q)
            err = (real_val-est_val)/real_val
            errs.append(err)
            ests.append(est_val)

        print(np.mean(errs))
        u_errs.append(np.mean(errs))
        u_stds.append(np.std(ests))
        u_coff_vars.append(np.std(ests)/np.mean(ests))
        #plt.hist(errs)
        #plt.show()

    qs = np.arange(0.1, 1, 0.1)
    plt.plot(qs, u_errs, label="percent bias")
    plt.plot(qs, u_coff_vars, label="coefficient of variation")
    plt.axhline(0, color="black")
    plt.axvline(0.5, color="black")
    plt.legend()
    plt.xlabel("Percentile")
    plt.show()


##################

for sampl in np.arange(35, 220,5):
    q = 0.9
    errs = []
    ests = []
    real_val = ppf_fn2(q)
    for _ in range(100000):
        x = rvs_fn2(sampl)
        est_val = estimate_percentile(x, q)
        err = (real_val-est_val)/real_val
        errs.append(err)
        ests.append(est_val)

    print(np.mean(errs))

    plt.hist(ests, bins=np.arange(0,14,1))
    plt.axvline(real_val, label="actual", color="yellow")
    plt.axvline(np.mean(ests), label="average", color="green")
    plt.axvline(estimate_percentile(ests, 1-q), label="percentile of estimates", color="orange")
    plt.axvline(estimate_percentile(ests, .5), label="median of estimates", color="purple")
    plt.legend()
    plt.savefig('plots/sample_' + str(sampl) + '.png')
    plt.close()
    print('processed sample size ' + str(sampl))

