import numpy as np
from scipy.stats import nbinom, binom_test, poisson
from stochproc.count_distributions.negative_binomial import NegativeBinomial
from scipy.stats import norm
import matplotlib.pyplot as plt
#TODO: Move plotting code to plots.py and remove dependency on pyray.
from pyray.plotting.matplot_utils import newline


class CompareTests():
    def __init__(self,r0=1.4, mu=0.7, rr=1.15, k=0.4):
        self.t=mu/r0
        self.r1=r0*rr
        self.r0=r0; self.mu=mu; self.rr=1.15; self.k=k    
        self.alpha_hats = np.concatenate((np.arange(0.000000000001,0.0099,0.0000001),
                                        np.arange(0.01,1.00,0.001), 
                                        np.arange(0.991,1.00,0.001)),axis=0)

    def plot_tests(self, n_sim=1000, n0=1035, n1=1035):
        c=self
        self.alphas_wald = np.zeros(len(self.alpha_hats))
        self.alphas_rate = np.zeros(len(self.alpha_hats))
        self.betas_wald = np.zeros(len(self.alpha_hats))
        self.betas_rate = np.zeros(len(self.alpha_hats))
        nb_null = NegativeBinomial(mu=c.mu,k=c.k,t=c.t)
        nb_alt = NegativeBinomial(mu=c.mu*c.rr,k=c.k,t=c.t)
        recall = 0
        for _ in range(n_sim):
            x_s = 0
            for i in range(n0):
                x_i = nb_null.rvs()
                #x_i = poisson.rvs(c.mu)
                x_s += x_i
            x_s_1 = 0
            for i in range(n1):
                x_i = nb_null.rvs()
                #x_i = poisson.rvs(c.mu)
                x_s_1 += x_i
            y_s = 0
            for j in range(n1):
                y_j = nb_alt.rvs()
                #y_j = poisson.rvs(c.mu*c.rr)
                y_s += y_j
            mu_est = x_s/n0
            p_val_wald_null = wald_tst(x_s,x_s_1,mu_est,n0,n1)
            p_val_rate_null = rate_tst(x_s,x_s_1,n0,n1)
            c.alphas_wald += (p_val_wald_null < c.alpha_hats)/n_sim
            c.alphas_rate += (p_val_rate_null < c.alpha_hats)/n_sim
            p_val_wald_alt = wald_tst(x_s,y_s,mu_est,n0,n1)
            p_val_rate_alt = rate_tst(x_s,y_s,n0,n1)
            c.betas_wald += 1/n_sim - (p_val_wald_alt < c.alpha_hats)/n_sim
            c.betas_rate += 1/n_sim - (p_val_rate_alt < c.alpha_hats)/n_sim
            recall += p_val_wald_alt<0.05
        ## This error rate should match with the paper.
        #print(type_2_errs/1000)
        plt.plot(c.alphas_wald,c.betas_wald,label='wald test')
        plt.plot(c.alphas_rate,c.betas_rate,label='rate test')
        p1 = [1,0]
        p2 = [0,1]
        newline(p1,p2)
        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.legend()
        plt.show()
        wald_5prct_ix = np.argmin((self.alphas_rate-0.05)**2)
        rate_5prct_ix = np.argmin((self.alphas_wald-0.05)**2)
        self.beta_5prct_rate = self.betas_rate[wald_5prct_ix]
        self.beta_5prct_wald = self.betas_wald[wald_5prct_ix]
        self.alpha_5prct_rate = self.alphas_rate[rate_5prct_ix]
        self.alpha_5prct_wald = self.alphas_wald[wald_5prct_ix]
        return recall/1000


def wald_tst(x_s,y_s,mu,n0,n1):
    c = CompareTests()
    r0_est = x_s/c.t/n0
    r1_est = y_s/c.t/n1
    tht = n1/n0
    var_null = ((1+tht)**2/(mu*tht)/(r0_est+r1_est*tht)+(1+tht)*c.k/tht)/n0
    #var_null = (4/mu/(r0_est+r1_est))/n0
    st_dev_null = np.sqrt(var_null)
    z_stat = (np.log(r1_est)-np.log(r0_est))/st_dev_null
    p_val = norm.sf(z_stat)
    return p_val


def rate_tst(x_s,y_s,n0=10,n1=10):
    p=n1/(n0+n1)
    p_val = binom_test(y_s,x_s+y_s,p,alternative='greater')
    return p_val

