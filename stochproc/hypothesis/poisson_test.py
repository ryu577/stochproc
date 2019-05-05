import numpy as np
from scipy.special import comb


class PoissonTest():
    @staticmethod
    def conditional_pmf(n,n2,t1,t2):
        """
        Calculates P(N_1=n_1|N=n) where N=N_1+N_2
        """
        t=t1+t2; n1=n-n2
        res = 0
        for i in range(n1):
            res += np.log(t1)-np.log(t)-\
                    np.log(n1-i)+np.log(n-i)
        for i in range(n2):
            res += np.log(t2)-np.log(t)-\
                np.log(n2-i)+np.log(n-n1-i)
        #return t1**n1*t2**n2/t**n/comb(n,n1)
        return np.exp(res)

    @staticmethod
    def p_value(n1,n2,t1,t2):
        n=n1+n2; t=t1+t2
        delta = n2/n-t2/t
        k_lo = int(np.ceil(delta*n+t2*n/t))
        p_val = 0
        for k in range(k_lo,int(n+1)):
            p_val += PoissonTest.conditional_pmf(n1+k,k,t1,t2)
        return p_val


