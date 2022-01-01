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
    return lomax.rvs(c=.9,size=n)

def rvs_fn5(n):
    return weibull_min.rvs(c=.5,size=n)

def ppf_fn1(q):
    return norm.ppf(q, 10, 1)

def ppf_fn2(q):
    return lognorm.ppf(q, 1, 0)

def ppf_fn3(q):
    return expon.ppf(q)

def ppf_fn4(q):
    return lomax.ppf(q,c=.9)

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


#def coeff_variation_and_bias(n=100, rvs_fn=rvs_fn2, ppf_fn=ppf_fn2):
n=15; rvs_fn=rvs_fn4; ppf_fn=ppf_fn4
u_errs = []
u_errs1 = []
u_stds = []
u_stds1 = []
u_coff_vars = []
u_medians = []
u_medians1 = []
u_mses = []
u_mses1 = []

qs = np.arange(0.3, .7, 0.03)

for q in qs:
    errs = []
    errs1 = []
    ests = []
    ests1 = []
    for _ in range(30000):
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
    u_mses.append(np.sqrt(np.var(ests)+np.mean(errs)**2))
    u_mses1.append(np.sqrt(np.var(ests1)+np.mean(errs1)**2))


plt.style.use('dark_background')

# Alternate plotting.
#fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
fig1, (ax1, ax3) = plt.subplots(2, 1)
fig2, (ax2, ax4) = plt.subplots(2, 1)


ax1.axhline(0, color="white")
ax1.axvline(0.5, color="white")
ax2.axhline(0, color="white")
ax2.axvline(0.5, color="white")
ax1.plot(qs, u_errs, label="Bias for linear interpolation strategy")
ax1.plot(qs, u_errs1, label="Bias for low bias strategy")
ax1.legend(prop={'size': 20})
ax2.plot(qs, u_stds, label="Standard deviation for linear interpolation strategy")
ax2.plot(qs, u_stds1, label="Standard deviation for low bias strategy")
ax2.legend(prop={'size': 20})
ax3.plot(qs, u_medians, label="DelMedian for linear interpolation strategy")
ax3.plot(qs, u_medians1, label="DelMedian for low bias strategy")
ax3.axhline(0, color="white")
ax3.axvline(0.5, color="white")
ax3.legend(prop={'size': 20})
ax4.plot(qs, u_mses, label="MSE for linear interpolation strategy")
ax4.plot(qs, u_mses1, label="MSE for low bias strategy")
ax4.axhline(0, color="white")
ax4.axvline(0.5, color="white")
ax4.legend(prop={'size': 20})
plt.xlabel("Percentile (q)")
ax2.tick_params(axis='x', labelsize=15)
ax4.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax2.tick_params(axis='y', labelsize=15)
ax4.tick_params(axis='y', labelsize=15)
ax1.tick_params(axis='y', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
plt.show()



###################################
####
plt.axhline(0, color="white")
plt.axvline(0.5, color="white")
plt.plot(qs, u_errs, label="Bias for linear interpolation strategy")
plt.plot(qs, u_errs1, label="Bias for low bias strategy")
plt.plot(qs, u_stds, label="Standard deviation for linear interpolation strategy")
plt.plot(qs, u_stds1, label="Standard deviation for low bias strategy")
plt.plot(qs, u_medians, label="DelMedian for linear interpolation strategy")
plt.plot(qs, u_medians1, label="DelMedian for low bias strategy")
plt.plot(qs, u_mses, label="MSE for linear interpolation strategy")
plt.plot(qs, u_mses1, label="MSE for low bias strategy")
plt.legend()
plt.xlabel("Percentile (q)")
plt.show()

