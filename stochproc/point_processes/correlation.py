import numpy as np
from scipy.stats import lomax, binom_test
import matplotlib.pyplot as plt
from stochproc.point_processes.renewal.lomax import lomax_renewal_stats, p_vals_lomax_renewal
from algorith.arrays.pt_process.overlap import critical_events
from algorith.arrays.pt_process.window import critical_interval
from hypothtst.alpha_beta_sim import rejectn_rate
from hypothtst.viz_utils.plots_on_pvals import plot_power_3alt, plot_pvals_3alt


## Some tests on correlated point processes
def tst_sim_renewal_process():
    k=0.8
    lmb=2.0
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=5000
    for _ in range(n_sim):
        intervals = lomax.rvs(c=k, scale=(1/lmb), size=800)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps = np.cumsum(intervals)
        n1 = sum((time_stamps>400) * (time_stamps<900))
        n2 = sum((time_stamps>900) * (time_stamps<1400))
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
        s_n1+=n1
        s_n2+=n2

    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("correlation: " +str(corln))

#####

def tst_sim_2():
    k=0.8
    lmb=2.0
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=5000
    for _ in range(n_sim):
        intervals1 = lomax.rvs(c=k, scale=(1/lmb), size=800)
        intervals2 = lomax.rvs(c=k, scale=(1/lmb), size=800)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps1 = np.cumsum(intervals1)
        time_stamps2 = np.cumsum(intervals2)
        n1 = sum((time_stamps1>400) * (time_stamps1<900))
        n2 = sum((time_stamps2>400) * (time_stamps2<900))
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
        s_n1+=n1
        s_n2+=n2
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("correlation: " +str(corln))


#####

def tst_sim_3(k=7.0,theta=0.5):
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=5000
    for _ in range(n_sim):
        intervals = lomax.rvs(c=k, scale=theta, size=2000)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps = np.cumsum(intervals)
        bi_furcator = np.random.choice(2,size=len(time_stamps))
        time_stamps1 = time_stamps[bi_furcator==1]
        time_stamps2 = time_stamps[bi_furcator==0]
        n1 = sum((time_stamps1>50) * (time_stamps1<90))
        n2 = sum((time_stamps2>50) * (time_stamps2<90))
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2
        s_n1+=n1
        s_n2+=n2
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("correlation: " +str(corln))

#####
## Check mean of Lomax renewal process

def lomax_renewal_correlation(k=2.0, theta=1.0):
    s_n1=0
    n_sim=5000
    for _ in range(n_sim):
        intervals = lomax.rvs(c=k, scale=theta, size=1200)
        #intervals = np.random.exponential(scale=1,size=1000)
        time_stamps = np.cumsum(intervals)
        #n1 = sum((time_stamps>100) * (time_stamps<200))
        n1 = sum(time_stamps<100)
        s_n1+=n1

    e_n1 = s_n1/n_sim

    print("simulated mean: " +str(e_n1))
    #print("actual mean-1: " +str(k*200/theta))
    print("actual mean-2: " +str((k-1)*200/theta))


#####
## Poisson mixture.. inducing correlation.

def mixed_poisson_correlation(k=2.0,theta=0.01):
    s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
    n_sim=3000
    for _ in range(n_sim):
        t=0; n1=0; n2=0
        lm = np.random.gamma(k,theta)
        #lm=1.2
        while t<130:
            t+=np.random.exponential(1/lm)
            toss = np.random.uniform()<0.5
            n1+=(t>100)*(t<110)*toss
            #n2+=(t>120)*(t<130)
            n2+=(t>100)*(t<110)*(1-toss)
        s_n1+=n1
        s_n2+=n2
        s_n1n2+=n1*n2
        s_n1_sq+=n1*n1
        s_n2_sq+=n2*n2

    e_n1 = s_n1/n_sim
    cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
    v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
    v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
    corln = cov/np.sqrt(v_n1*v_n2)
    print("Correlation: " + str(corln))
    print("Simulated mean: " + str(2*e_n1))
    print("Theoretical mean:" + str(k*10*theta))



## Verification code that Lomax is same as mixing exponentials:
# from survival.distributions.lomax import lomax_exponmix

