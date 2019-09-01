import numpy as np
from distributions.weibull import Weibull
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
    def rvs2(k=0.4,lmb=1.0,durtn=20.0,size=1000):
        """
        Same as rvs (which is more explicit and maintainable), 
        just including this one here to test for speed.
        """
        return np.array([sum(np.cumsum(Weibull.samples_(k,lmb,size=100))<durtn) \
            for _ in range(1000)])
    
    def __init__(self,k,lmb,durtn):
        self.k=k; self.lmb=lmb; self.durtn=durtn
    
    def rvs(self):
        return InterarrivalWeibull.rvs_s(self.k,self.lmb,self.durtn)


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
    plt.legend()
    plt.show()


