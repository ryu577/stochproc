import numpy as np
from scipy.stats import poisson

## We can assume that the number of VMs on a node 
#  will follow a binomial distribution.

class CompoundPoisson():
    @staticmethod
    def rvs(binom_n=23,binom_p=0.2):
        N = np.random.poisson(10)
        rv = 0
        for _ in range(N):
            vms = np.random.binomial(binom_n,binom_p)
            rv += vms
        return rv

binom_n=23; binom_p=0.2
rvs = []
for i in range(50000):
    rvs.append(CompoundPoisson.rvs(binom_n, binom_p))

rvs = np.array(rvs)

np.mean(rvs)
measured_var = np.var(rvs)

binom_e_ysq = binom_n*binom_p*(1-binom_p) + (binom_n*binom_p)**2

theoretic_var = binom_e_ysq*10

print("Measured variance:" + str(measured_var))
print("Theoretical variance:" + str(theoretic_var))

