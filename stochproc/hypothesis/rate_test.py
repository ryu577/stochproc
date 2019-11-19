import numpy as np
from scipy.special import comb
from scipy.stats import binom_test, poisson, binom
from stochproc.hypothesis.binom_test import binom_tst_beta
from scipy import optimize


class UMPPoisson(object):
    @staticmethod
    def beta_on_poisson(t1=25,t2=25,lmb_base=12,alpha=0.05,effect=3,
                        thresh=None,n=10000):
        """
        Obtains the beta (false negative rate) given the observation
        durations for treatment and control and hypothesis test to simulate.
        args:
            t1: VM-duration observed in treatment.
            t2: VM-duration observed in control.
            fn: The test to generate simulated p_value.
                for example: simulate_binned_t_test, simulate_rateratio_test.
        """
        if thresh is None:
            thresh = np.array([alpha])
        ## Find all betas at various values of alpha.
        betas = 1-est_rejection_rate(lmb1=lmb_base,lmb2=lmb_base+effect,t1=t1,t2=t2,
                                    fn=UMPPoisson.poisson_one_sim,thresh=thresh,n=n)
        # Select the beta at the alpha level that gives us 5% false positive rate.
        ##TODO: Change this to closest instead of exact match.
        beta = betas[thresh==alpha][0]
        return beta

    @staticmethod
    def beta_on_poisson_closed_form(t1=25,t2=25,lmb_base=12,effect=3,alpha=0.05):
        poisson_mu = lmb_base*t1+(lmb_base+effect)*t2
        beta = 0.0; prob_mass = 0.0
        p_null=t1/(t1+t2)
        mu_1 = t1*(lmb_base+effect); mu_2 = t2*lmb_base
        p_alt = mu_1/(mu_1+mu_2)
        int_poisson_mu = int(poisson_mu); pmf = 1.0
        while pmf > 1e-7 and int_poisson_mu>=0:
            pmf = poisson.pmf(int_poisson_mu,poisson_mu)
            prob_mass += pmf
            beta += pmf*binom_tst_beta(p_null,p_alt,int_poisson_mu,alpha)
            if np.isnan(beta):
                break
            int_poisson_mu -= 1

        int_poisson_mu = int(poisson_mu)+1; pmf=1.0

        while pmf > 1e-7:
            pmf = poisson.pmf(int_poisson_mu,poisson_mu)
            prob_mass += pmf
            beta += pmf*binom_tst_beta(p_null,p_alt,int_poisson_mu,alpha)
            int_poisson_mu += 1
        return beta, prob_mass

    @staticmethod
    def poisson_one_sim(lmb1,t1,lmb2,t2,alternative='greater'):
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
        p_val = binom_test(n2,n1+n2,t2/(t1+t2),alternative=alternative)
        return p_val


def p_n1(t1, t2, n1, n2):
    n=n1+n2; t=t1+t2
    return t1**n1*t2**n2/(t**n*comb(n,n1))


def rateratio_test(n1,t1,n2,t2,scale=1.0):
    n2, n1 = n2/scale, n1/scale
    p_val = binom_test(n2,n1+n2,t2/(t1+t2),alternative='greater')
    return p_val


def est_rejection_rate(lmb1=12.0, lmb2=12.0,
                        t1=2.5, t2=2.5, n=10000,
                        thresh=np.arange(0.001,1.0,0.01),
                        fn=UMPPoisson.poisson_one_sim):
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


def bake_time(t1=25,
                    lmb_base=12,alpha=0.05,
                    beta=0.05,effect=3,n=1000):
    t2=1.0; beta_tmp=1.0
    betas = []
    while beta_tmp>beta:
        beta_tmp = UMPPoisson.beta_on_poisson(t1=t1,t2=t2,\
                    lmb_base=lmb_base,
                    alpha=alpha,effect=effect,n=n)
        betas.append(beta_tmp)
        t2+=1
    return t2, np.array(betas)


def bake_time_v2(t1=25,
                    lmb_base=12,alpha=0.05,
                    beta=0.05,effect=3):
    t2=1.0; beta_tmp=1.0
    betas = []
    while beta_tmp>beta:
        beta_tmp = UMPPoisson.beta_on_poisson_closed_form(t1=t1,t2=t2,\
                    lmb_base=lmb_base,
                    alpha=alpha,effect=effect)[0]
        betas.append(beta_tmp)
        t2+=1
    return t2, np.array(betas)


def bake_time_v3(t1=25,
                    lmb_base=12,alpha=0.05,
                    beta=0.05,effect=3):
    fn = lambda t2: UMPPoisson.beta_on_poisson_closed_form(t1=t1,t2=t2,\
                        lmb_base=lmb_base,
                        alpha=alpha,effect=effect)[0]-beta
    if fn(100)*fn(1)>0:
        return 100
    root = optimize.bisect(fn,1,200)
    #root = optimize.root(fn,x0=5).x[0]
    return root


##t1 and t2 are in 100-VM-days
### lmb_base: 1 failure per 100-VM-days.
## 10 nodes per hw and 10 VMs per node. So, 100 VMs per day.

UMPPoisson.beta_on_poisson_closed_form(t1=1.0,t2=1.0,\
                        lmb_base=20,
                        alpha=0.1,effect=20)

## We need 20 events per 100-VM-days.

n=660
UMPPoisson.beta_on_poisson_closed_form(t1=n/10,t2=n/10,\
                        lmb_base=20,
                        alpha=0.1,effect=20*.1)

