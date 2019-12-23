import numpy as np
from scipy.stats import nbinom


class NegativeBinomial():
    @staticmethod
    def rvs_s(n,p,size=1000):
        return nbinom.rvs(n, p, size=size)

    def __init__(self, mu, k, t):
        ## mu and k is the parametrization 
        # used by Zhu and Lakkis.
        self.mu=mu; self.k=k; self.t=t
        self.m=1/k; self.theta = self.m*self.t/self.mu
        self.p = 1/(1+self.k*self.mu)

    def rvs(self):
        return nbinom.rvs(self.m,self.p)


def p_n1_pl_n2(n,theta,m,t1,t2):
    summ = 0
    p1 = theta/(t1+theta)
    p2 = theta/(t2+theta)
    for j in range(n+1):
        summ += nbinom.pmf(j,m,p1)*\
                nbinom.pmf(n-j,m,p2)
    return summ


if __name__ == '__main__':
    t1=1; t2=2
    theta = 0.5; m = 3
    summn = p_n1_pl_n2(5,theta,m,t1,t2)
    p = theta/(theta+t1+t2)
    actual_p = nbinom.pmf(5,m,p)
    print(actual_p-summn)


