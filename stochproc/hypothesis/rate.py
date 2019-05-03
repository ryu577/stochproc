import numpy as np
from scipy.stats import poisson
from scipy.optimize import root, bisect


def pois_diff_sf(d,lmb1,lmb2,t=10e3/4/100,s=15e3/4/100,terms=1000):
    """
    Calculates the survival function of the random variable:
    Poisson(lmb1*t)/t-Poisson(lmb2*s)/s at the value, d.
    """
    ans = 0
    #Do the summation for terms around the mean.
    mean = int(lmb2*s)
    #Calculate the double summation.
    term=1e-3; i=mean
    while term>1e-8:
        j = np.floor(t*(d+i/s))
        term = poisson.pmf(i,lmb1*s)*\
               poisson.sf(j,lmb2*t)
        ans+=term
        i+=1
    term=1e-3; i=mean-1
    while term>1e-8 and i>0:
        j = np.floor(t*(d+i/s))
        term = poisson.pmf(i,lmb1*s)*\
               poisson.sf(j,lmb2*t)
        ans+=term
        i-=1
    return ans


def pois_diff_surv_inv(p, lmb1, lmb2, t1=10e3/4/100, t2=15e3/4/100):
    """
    The inverse of the survival function of distribution
    defining the difference of two scaled poisson distributions,
    Poisson(lmb1*t1)/t1-Poisson(lmb2*t2)/t2
    """
    sf = lambda d: pois_diff_sf(d,lmb1,lmb2,t1,t2)-p
    return bisect(sf,-50,50)


def beta(lmb, effect, t1, t2, alpha):
    """
    Calculates the beta (type-2 error; 
    P(alternate is true but fail to reject null))
    1-beta becomes the power of the test.
    """
    s_inv = pois_diff_surv_inv(alpha,lmb,lmb,t1,t2)
    return cdf_alternate(s_inv,lmb,effect,t1,t2)


def pois_diff_cdf(d,lmb,t,s,nsim=100000):
    """
    Returns the probability that N_1/t-N_2/s<d
    where N_1~Pois(lmb*t) and N_2~Pois(lmb*s) using
    simulation.
    """
    n1 = poisson.rvs(lmb*t,size=nsim)
    n2 = poisson.rvs(lmb*s,size=nsim)
    return sum(n1/t-n2/s <= d)/nsim


def cdf_alternate(z, lmb, effect, t1=10e3/4/100, t2=15e3/4/100):
    """
    The CDF of the random variable Poisson(lmb1*t1)/t1-Poisson(lmb2*t2)/t2
    where lmb1 = lmb and lmb2 = lmb+effect.
    """
    return 1-pois_diff_sf(z,lmb,lmb+effect,t1,t2)


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
    p_val2 = pois_diff_sf(d[0],lmb_mix[0],lmb_mix[0],t,s)
    return d[0], p_val1, p_val2


###################################

def confusion_matrix(t1=10e3/4/100, t2=15e3/4/100):
    ## Broken method. Don't use.
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
        p_accept_alternate = pois_diff_sf(d_stat,lmb_hat,lmb_hat,t1,t2)
        confusion[confusion_term,] += np.array([p_accept_alternate, 1-p_accept_alternate])
    return confusion


###################################
## Plotting.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def collect_data():
    lmb=20.0
    d=2.0
    res=np.zeros((20,20))
    for t in range(20):
        for s in range(20):
            res[t,s] = pois_diff_sf(d,lmb,lmb,t+1,s+1)
    return res


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


def plot_alpha_beta():
    alphas = np.arange(0,1,.1)
    betas = np.array([beta(10.0,1,25,25,i) for i in alphas])
    plt.plot(alphas,1-betas)
    plt.show()

