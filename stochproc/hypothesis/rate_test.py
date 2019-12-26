import numpy as np
from scipy.special import comb
from scipy.stats import binom_test, poisson, binom, nbinom
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
    def beta_on_poisson_closed_form(t1=25,t2=25,\
                lmb_base=12,effect=3,alpha=0.05):
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
    def beta_on_poisson_closed_form2(t1=25,t2=25,\
                lmb_base=12,effect=3,alpha=0.05):
        beta=0; n=0
        beta_n=0; beta_del=0
        p=lmb_base*t1/(lmb_base*t1+(lmb_base+effect)*t2)
        q=t1/(t1+t2)
        mu_1 = t1*(lmb_base+effect); mu_2 = t2*lmb_base
        poisson_mu = lmb_base*t1+(lmb_base+effect)*t2
        int_poisson_mu = int(poisson_mu)
        n = int_poisson_mu-1
        while beta_del > 1e-9 or n==int_poisson_mu-1:
            n+=1
            surv_inv = int(binom.isf(alpha,n,q))
            beta_del=0
            for j in range(surv_inv+1):
                beta_n = poisson.pmf(j,(lmb_base+effect)*t2)*poisson.pmf(n-j,lmb_base*t1)
                beta_del+=beta_n
                beta += beta_n
        n = int_poisson_mu
        while beta_del > 1e-9 or n==int_poisson_mu:
            n-=1
            surv_inv = int(binom.isf(alpha,n,q))
            beta_del=0
            for j in range(surv_inv+1):
                beta_n = poisson.pmf(j,(lmb_base+effect)*t2)*poisson.pmf(n-j,lmb_base*t1)
                beta_del+=beta_n
                beta += beta_n
        return beta

    @staticmethod
    def beta_on_poisson_closed_form3(t1=25,t2=25,\
                lmb_base=12,effect=3):
        ## This method is only for alpha=0.5
        poisson_mu = (lmb_base+effect)*t2
        poisson_mu_base = lmb_base*t1
        prob_mass = 0.0
        int_poisson_mu = int(poisson_mu); pmf = 1.0
        beta = 0
        while pmf > 1e-7 and int_poisson_mu>=0:
            pmf = poisson.pmf(int_poisson_mu,poisson_mu)
            prob_mass += pmf
            beta += pmf*poisson.sf(int_poisson_mu-1,poisson_mu_base)
            int_poisson_mu -= 1
        int_poisson_mu = int(poisson_mu)+1; pmf=1.0
        while pmf > 1e-7:
            pmf = poisson.pmf(int_poisson_mu,poisson_mu)
            prob_mass += pmf
            beta += pmf*poisson.sf(int_poisson_mu-1,poisson_mu_base)
            int_poisson_mu += 1
        return beta, prob_mass

    @staticmethod
    def beta_on_negbinom_closed_form(t1=25,t2=25,\
                theta_base=10,m=100.0,deltheta=3,alpha=0.05,cut_dat=1e4):
        del_lmb = m*deltheta/theta_base/(theta_base-deltheta)
        return UMPPoisson.beta_on_negbinom_closed_form2(t1,t2,theta_base,m,del_lmb,alpha,cut_dat)

    @staticmethod
    def beta_on_negbinom_closed_form2(t1=25,t2=25,\
                theta_base=10,m=100.0,effect=3,alpha=0.05,cut_dat=1e4):
        beta=0; n=0
        beta_n=0; beta_del=0
        q=t1/(t1+t2)
        lmb_base = m/theta_base
        mu_1 = t1*(lmb_base+effect); mu_2 = t2*lmb_base
        p1 = theta_base/(theta_base+t1)
        del_theta = theta_base**2*effect/(m+theta_base*effect)
        theta2=theta_base-del_theta
        p2 = theta2/(t2+theta2)
        poisson_mu = lmb_base*t1+(lmb_base+effect)*t2
        int_poisson_mu = int(poisson_mu)
        n = int_poisson_mu-1
        dels1 = []; ns1=[]
        if effect == 0:
            nbinom_s1={}; nbinom_s2 = nbinom_s1
        else:
            nbinom_s1={}; nbinom_s2={}
        while (beta_del > 1e-9 or n==int_poisson_mu-1):
            n+=1
            if n-int_poisson_mu>cut_dat:
                break
            surv_inv = int(binom.isf(alpha,n,q))
            beta_del=0
            for j in range(surv_inv+1):
            #for j in range(n+1):
                if j in nbinom_s1:
                    nb1 = nbinom_s1[j]
                else:
                    nb1 = nbinom.pmf(j,m,p2)
                    nbinom_s1[j] = nb1
                if n-j in nbinom_s2:
                    nb2 = nbinom_s2[n-j]
                else:
                    nb2 = nbinom.pmf(n-j,m,p1)
                    nbinom_s2[n-j] = nb2
                beta_n = nb1*nb2
                beta_del+=beta_n
                beta += beta_n
            dels1.append(beta_del); ns1.append(n)
        n = int_poisson_mu
        dels2 = []; ns2=[]
        while beta_del > 1e-9 or n==int_poisson_mu:
            n-=1
            if int_poisson_mu-n>cut_dat:
                break
            surv_inv = int(binom.isf(alpha,n,q))
            beta_del=0
            for j in range(surv_inv+1):
            #for j in range(n+1):
                if j in nbinom_s1:
                    nb1 = nbinom_s1[j]
                else:
                    nb1 = nbinom.pmf(j,m,p2)
                    nbinom_s1[j] = nb1
                if n-j in nbinom_s2:
                    nb2 = nbinom_s2[n-j]
                else:
                    nb2 = nbinom.pmf(n-j,m,p1)
                    nbinom_s2[n-j] = nb2
                beta_n = nb1*nb2
                beta_del+=beta_n
                beta += beta_n
            dels2.append(beta_del); ns2.append(n)
        dels1 = np.array(dels1); dels2 = np.array(dels2); dels2 = dels2[::-1]
        ns1 = np.array(ns1); ns2 = np.array(ns2); ns2 = ns2[::-1]
        ns = np.concatenate((ns2,ns1),axis=0)
        dels = np.concatenate((dels2,dels1),axis=0)
        return beta, dels, ns, int_poisson_mu

    @staticmethod
    def beta_on_negbinom_closed_form3(t1=25,t2=25,\
                theta_base=10,m=100.0,deltheta=3):
        """
        This method only works for alpha=0.5.
        """
        if deltheta > theta_base:
            #TODO: Replace this with exception.
            print("deltheta must be smaller than theta.")
            return
        theta_alt = theta_base-deltheta
        neg_binom_ix = 0
        p2 = theta_alt/(t2+theta_alt)
        p1 = theta_base/(t1+theta_base)
        mode1 = int(p1*(m-1)/(1-p1))
        beta = 0; del_beta = 1
        while del_beta>1e-7 or neg_binom_ix<mode1 or neg_binom_ix<1000:
            del_beta = nbinom.pmf(neg_binom_ix,m,p2)*nbinom.sf(neg_binom_ix-1,m,p1)
            beta += del_beta
            neg_binom_ix+=1
        return beta


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


def experiments():
    ##t1 and t2 are in 100-VM-days
    ### lmb_base: 1 failure per 100-VM-days.
    ## 10 nodes per hw and 10 VMs per node. So, 100 VMs per day.

    UMPPoisson.beta_on_poisson_closed_form(t1=1.0,t2=1.0,\
                            lmb_base=20,\
                            alpha=0.1,effect=20)

    ## We need 20 events per 100-VM-days.

    n=660
    UMPPoisson.beta_on_poisson_closed_form(t1=n/10,t2=n/10,\
                            lmb_base=20,\
                            alpha=0.1,effect=20*.1)


    UMPPoisson.beta_on_poisson_closed_form2(t1=1.0,t2=1.0,\
                            lmb_base=20,\
                            alpha=0.1,effect=20)


    import matplotlib.pyplot as plt

    res=UMPPoisson.beta_on_negbinom_closed_form2(t1=200,t2=200,cut_dat=1000)
    plt.plot(res[2],res[1])
    plt.axvline(res[3])
    plt.show()


import matplotlib.pyplot as plt

def binom_partial_sum(n,p=.5):
    b_sum=0
    for j in range(int(n/1.5)+1):
        b_sum+=comb(n,j)*(1+p)**j
    return b_sum/(2+p)**n

if __name__ == '__main__':
    sums = np.array([binom_partial_sum(i,p=0.4) for i in range(11,501,2)])
    plt.plot(np.arange(11,501,2),sums)

