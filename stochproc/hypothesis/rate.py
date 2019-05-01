import numpy as np
from scipy.stats import poisson


def rate_hypothesis_test(lmb=12.0, mu=14.5, t=10e3/4/100, s=15e3/4/100):
    """
    Calculates the simulated p-value for the hypothesis test.
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
    p_val1 = 1-pois_diff_cdf(d[0],lmb_mix[0],t,s)
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


def pois_diff_sf(d,lmb,t,s):
    ans = 0
    mean = int(lmb*t)
    for i in range(mean-1000,mean+1000):
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


