import numpy as np
from scipy.stats import poisson, logser

## We can assume that the number of VMs on a node 
#  will follow a binomial distribution.

class CompoundPoisson():
    @staticmethod
    def rvs_s(lmb=10,binom_n=23,binom_p=0.2,compound='binom'):
        N = np.random.poisson(lmb)
        rv = 0
        for _ in range(N):
            if compound == 'binom':
                vms = np.random.binomial(binom_n,binom_p)
            elif compound == 'binom_pl1':
                vms = np.random.binomial(binom_n,binom_p)+1
            else:
                vms = logser.rvs(.8)
            rv += vms
        return rv
    
    @staticmethod
    def rvs_s_1(compound_rvs, lmb=10):
        n = np.random.poisson(lmb)
        rv=0
        for _ in range(n):
            rv += compound_rvs()
        return rv

    def __init__(self,lmb=10,comp=logser):
        self.lmb=lmb
        self.comp = comp
    
    def rvs(self):
        #return CompoundPoisson.rvs_s(self.n,self.p)
        return 1

    def cdf(self,x):
        return sum(np.array([self.rvs() \
                    for i in range(10000)]) < x)


def verify_variance():
    binom_n=23; binom_p=0.2
    rvs = []
    for _ in range(50000):
        rvs.append(CompoundPoisson.rvs_s(binom_n, binom_p))

    rvs = np.array(rvs)

    np.mean(rvs)
    measured_var = np.var(rvs)

    binom_e_ysq = binom_n*binom_p*(1-binom_p) + (binom_n*binom_p)**2

    theoretic_var = binom_e_ysq*10

    print("Measured variance:" + str(measured_var))
    print("Theoretical variance:" + str(theoretic_var))

