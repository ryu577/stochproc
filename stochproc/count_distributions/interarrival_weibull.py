import numpy as np
from stochproc.distributions.weibull import Weibull
import matplotlib.pyplot as plt
from scipy.special import gamma


class InterarrivalWeibull():
    @staticmethod
    def rvs_s(k=0.4,lmb=1.0,durtn=20.0,size=1000):
        cnts = []
        for _ in range(size):
            w_rvs = Weibull.samples_(k,lmb,size=100)
            w_arrivals = np.cumsum(w_rvs)
            # TODO: Need to ensure that durtn is less than the cumsum of the Weibulls
            cnt = sum(w_arrivals<durtn)    
            cnts.append(cnt)
        return np.array(cnts)

    @staticmethod
    def rvs1_s(k=0.4,lmb=1.0,durtn=20.0):
        w_rvs = Weibull.samples_(k,lmb,size=1)[0]; cnt=0
        while w_rvs < durtn:
            w_rvs += Weibull.samples_(k,lmb,size=1)[0]
            cnt +=1
        return cnt

    @staticmethod
    def rvs2_s(k=0.4,lmb=1.0,durtn=20.0,size=1000):
        """
        Same as rvs (which is more explicit and maintainable), 
        just including this one here to test for speed.
        """
        return np.array([sum(np.cumsum(Weibull.samples_(k,lmb,size=100))<durtn) \
            for _ in range(1000)])
    
    def __init__(self,k,lmb,durtn):
        """
        k<1 means decreasing haz rate, positively correlated events and overdispersed (var>mean)
        k>1 means increasing haz rate, negatively correlated events and underdispersed (var<mean)
        """
        self.k=k; self.lmb=lmb; self.durtn=durtn
        self.weib_mean = self.lmb*gamma(1+1/self.k)
        self.naive_calc_mean = self.durtn/self.weib_mean
    
    def rvs(self,size=10):
        return InterarrivalWeibull.rvs_s(self.k,self.lmb,self.durtn,size=size)
    
    def rvs1(self):
        return InterarrivalWeibull.rvs1_s(self.k,self.lmb,self.durtn)


# Q - how does the mean change as we change the parameters of the Weibull?
def plot_mean_with_k():
    actual_means = []; expctd_means = []
    for k in np.arange(0.1, 4.0, 0.1):
        actual_mean = np.mean(InterarrivalWeibull.rvs_s(k,durtn=20.0))
        w_mean = gamma(1+1/k)
        expctd_mean = 20.0/w_mean
        actual_means.append(actual_mean)
        expctd_means.append(expctd_mean)
    plt.plot(np.arange(0.1,4.0,0.1),actual_means,label='actual')
    plt.plot(np.arange(0.1,4.0,0.1),expctd_means,label='expected')
    plt.xlabel('Shape parameter of Weibull')
    plt.ylabel('Number of events in 20 units of time')
    plt.axvline(1.0)
    plt.legend()
    plt.show()


