import numpy as np
from scipy.stats import poisson


def rate_hypothesis_test_1(lmb=12.0, mu=14.5, t=10e3/4/100, s=15e3/4/100):
    """
    Calculates the p-value for hypothesis test-1 based on rate difference.
    args:
        lmb: The rate in interruptions per 100 VM years for first group.
        mu: The rate in interruptions per 100 VM years for second group.
        t: The time for which first group runs in 100 VM years. Default assuming
           10K VMs for 3 months; time in 100 VM years.
        s: The time for which the second group runs in 100 VM years.
           Default assuming 15K VMs for 3 months; time in 100 VM years.
    """
    pois1 = poisson.rvs(lmb*t,size=1)
    pois2 = poisson.rvs(mu*s,size=1)
    # Get the estimated rates.
    lmb_est = pois1/t
    mu_est = pois2/s
    lmb_mix = (pois1+pois2)/(s+t)
    d = mu_est-lmb_est
    ## Uses simulation
    p_val1 = 1-pois_diff_cdf(d[0],lmb_mix[0],t,s)
    ## Uses the summation
    p_val2 = pois_diff_sf(d[0],lmb_mix[0],t,s)
    return d[0], p_val1, p_val2


def pois_diff_cdf(d,lmb,t,s,nsim=100000):
    """
    Returns the probability that N_1/t-N_2/s<d
    where N_1~Pois(lmb*t) and N_2~Pois(lmb*s)
    """
    n1 = poisson.rvs(lmb*t,size=nsim)
    n2 = poisson.rvs(lmb*s,size=nsim)
    return sum(n1/t-n2/s < d)/nsim


def pois_diff_sf(d,lmb,t,s,terms=1000):
    ans = 0
    mean = int(lmb*t)
    for i in range(mean-terms,mean+terms):
        j = np.floor(t*(d+i/s))
        ans += poisson.pmf(i,lmb*s)*\
               poisson.sf(j,lmb*t)
    return ans


def collect_data():
    lmb=20.0
    d=2.0
    res=np.zeros((20,20))
    for t in range(20):
        for s in range(20):
            res[t,s] = pois_diff_sf(d,lmb,t+1,s+1)
    return res


def confusion_matrix(t1=10e3/4/100, t2=15e3/4/100):
    confusion = np.zeros((2,2))
    for _ in range(10000):
        lmb1 = lmb2 = 12.0
        confusion_term = 0
        if np.random.uniform() > 0.5:
            lmb2 += 0.2
            confusion_term = 1
        n1 = poisson.rvs(lmb1*t1)
        n2 = poisson.rvs(lmb2*t2)
        lmb1_hat = n1/t1
        lmb2_hat = n2/t2
        lmb_hat = (n1+n2)/(t1+t2)
        d_stat = lmb2_hat-lmb1_hat
        p_accept_alternate = pois_diff_sf(d_stat,lmb_hat,t1,t2)
        confusion[confusion_term,] += np.array([p_accept_alternate, 1-p_accept_alternate])
    return confusion


## Plotting.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def make_plot():
    ts = np.arange(20)
    ss = np.arange(20)
    # Make data.
    X = ts
    Y = ss
    X, Y = np.meshgrid(X, Y)
    res = collect_data()
    Z = res
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


