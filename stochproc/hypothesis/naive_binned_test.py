import numpy as np
from scipy import stats
from scipy.stats import poisson
from scipy.stats import binom_test
from stochproc.hypothesis.rate import *

#np.set_printoptions(linewidth=np.inf)

def simulate_rateratio_test(lmb1,t1,lmb2,t2):
    """
    Simulates data from two Poisson distributions
    and finds the p-value for a one-sided test.
    args:
        lmb1: The failure rate for first population.
        t1: The time observed for first population.
        lmb2: The failure rate for second population.
        t2: The time observed for second population
    """
    n1 = poisson.rvs(lmb1*t1)
    n2 = poisson.rvs(lmb2*t2)
    p_val = binom_test(n2,n1+n2,t2/(t1+t2),alternative='greater')
    return p_val

def simulate_binned_t_test(lmb1,t1,lmb2,t2,bin_size=1.0):
    """
    Simulates data from two Poisson distributions,
    bins them as per the bin-size and finds the p-value
    by passing the binned AIR estimate vectors to a two-
    sided t-test.
    args:
        lmb1: The failure rate for first population.
        t1: The time observed for first population.
        lmb2: The failure rate for second population.
        t2: The time observed for second population
        bin_size: The bins into which data is partitioned.
    """
    num_bins1 = int(t1/bin_size)
    num_bins2 = int(t2/bin_size)
    if num_bins1 < 2 or num_bins2<2:
        print("Not enough bins!")
        return
    n1 = poisson.rvs(lmb1*t1/num_bins1,size=num_bins1)
    n2 = poisson.rvs(lmb2*t2/num_bins2,size=num_bins2)
    mean1 = np.mean(n1/bin_size); std1=np.std(n1/bin_size)
    mean2 = np.mean(n2/bin_size); std2=np.std(n2/bin_size)
    p_val = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=20, \
                        mean2=mean2, std2=std2, nobs2=20, \
                        equal_var=False).pvalue/2
    return p_val

def est_rejection_rate(lmb1=12.0, lmb2=12.0,
                        t1=2.5, t2=2.5, n=100000,
                        thresh=np.arange(0.001,1.0,0.01),
                        fn=simulate_rateratio_test):
    """
    Given various values of alpha, gets the percentage of time
    the second sample is deemed to have a greater AIR than the
    first sample.
    args:
        lmb1: The failure rate of the first population.
        lmb2: The failure rate of the second population.
        t1: The time data is collected for first population.
        t2: The time data is collected for the second population.
        n: The number of simulations.
        thresh: The alpha levels.
        fn: The test to generate simulated p_value.
            for example: simulate_binned_t_test, simulate_rateratio_test.
    """
    reject_rate=np.zeros(len(thresh))
    for _ in range(n):
        #n1 is control, n2 is treatment.
        p_val = fn(lmb1,t1,lmb2,t2)
        reject_rate+=(p_val<thresh)
    return reject_rate/n


def get_beta(t1=25,t2=25,fn=simulate_binned_t_test,lmb_base=12,alpha=0.05,effect=3):
    """
    Obtains the beta (false negative rate) given the observation
    durations for treatment and control and hypothesis test to simulate.
    args:
        t1: VM-duration observed in treatment.
        t2: VM-duration observed in control.
        fn: The test to generate simulated p_value.
            for example: simulate_binned_t_test, simulate_rateratio_test.
    """
    alphas = np.arange(0.001,1.0,0.01)
    ## For small samples, the actual false positive rate differs from the alphas we set.
    # so, we choose the alpha that gives us a false positive rate of 5%.
    real_alphas = est_rejection_rate(lmb1=lmb_base,lmb2=lmb_base,t1=t1,t2=t2,fn=fn)
    errs = (real_alphas-alpha)**2
    if min(errs) > 0.001:
        raise Exception("Too far!")
    set_alpha = alphas[np.argmin(errs)]
    set_alpha_idx = np.argmin(errs)
    ## Find all betas at various values of alpha.
    betas = 1-est_rejection_rate(lmb1=lmb_base,lmb2=lmb_base+effect,t1=t1,t2=t2,fn=fn)
    # Select the beta at the alpha level that gives us 5% false positive rate.
    beta = betas[set_alpha_idx]
    return beta


def get_ctrl_sample(t1=25,fn=simulate_rateratio_test,
                    lmb_base=12,alpha=0.05,beta=0.05,effect=3):
    t2=1.0; beta_tmp=1.0
    while beta_tmp>beta:
        beta_tmp = get_beta(t1=t1,t2=t2,fn=fn,lmb_base=lmb_base,alpha=alpha,effect=effect)
        t2+=1
    return t2


############################

def compare_tests(n=1e4, alpha=np.array([.01,.25,.3,.4,.45,.5]),
                 lmb=12.0, mu=12.0):
    cnt=np.zeros(len(alpha)); cnt1=np.zeros(len(alpha))
    for _ in range(int(n)):
        t=10e3/4/1000; s=10e3/4/1000
        n1s = poisson.rvs(lmb*t,size=20)
        n2s = poisson.rvs(mu*s,size=20)
        rate1 = n1s/t
        rate2 = n2s/s
        n1 = sum(n1s); n2 = sum(n2s)
        d = n2/(20*s)-n1/(20*t)
        lmb_mix = (n1+n2)/(t+s)/20
        p_val2 = pois_diff_sf(d,lmb_mix,lmb_mix,t,s)
        #if p_val2 < alpha:# and n2/s>n1/t:
        cnt1 += p_val2<alpha
        mean1 = np.mean(rate1); std1=np.std(rate1)
        mean2 = np.mean(rate2); std2=np.std(rate2)
        #if mean2>mean1:
        t_score = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=20, \
                            mean2=mean2, std2=std2, nobs2=20, \
                            equal_var=False)
        #if t_score.pvalue/2 < alpha:
        cnt += t_score.pvalue/2 < alpha
    print(cnt/n)
    print(cnt1/n)


def plot_alpha_beta_curves():
    xs1 = np.array([0,0.000181818,    0.000363636,   0.001818182,    0.010590909,    0.038863636,    0.041545455 ,   0.127454545 ,   0.128045455  ,  0.283136364   , 0.283363636 ,   0.509136364,1])
    ys1=np.array([1,0.656846154,    0.441192308,    0.309461538,    0.103076923,    0.097307692 ,   0.036307692  ,  0.009846154 ,   0.009769231,    0.0015 ,   0.0015 ,   0.000230769,0])
    plt.plot(xs1,ys1,label='Poisson difference test')
    xs2 = np.array([0,0.022909091,	0.044363636,	0.064681818,	0.086727273	,0.207772727,	0.229,	0.248818182,	0.268636364,	0.309045455,	0.349818182,	0.408409091,	0.446409091,	0.504272727,	0.603045455,	0.661181818,	0.702045455,	0.798363636,	0.839954545,1])
    ys2 = np.array([1,0.147846154,	0.089769231,	0.064, 0.049807692,	0.016576923,	0.014538462,	0.012807692,	0.011038462,	0.008230769,	0.006307692,	0.004807692,	0.003730769,	0.002692308,	0.001346154,	0.001,	0.000846154,	0.000461538,	0.000307692,0])
    plt.plot(xs2,ys2,label='Time binned Welch test')
    plt.xlabel('false positive rate (alpha)')
    plt.ylabel('false negative rate (beta)')
    plt.legend()
    plt.show()


#1-est_rejection_rate_binned(lmb1=12.0,lmb2=15.0,thresh=0.038)
#1-est_rejection_rate(lmb1=12.0,lmb2=15.0,t1=25.0,t2=25.0,thresh=0.0545,n=int(1e5))


def get_full_betas_grid():
    binom_tst=np.zeros((20,20))
    for i in range(3,20):
        for j in range(i,20):
            binom_tst[i,j] = get_beta(t1=i,t2=j,
                    fn=simulate_rateratio_test)
            print(str(binom_tst[i,j]) + ",")
        #print()
    np.savetxt('binom_tst1.csv',binom_tst)

