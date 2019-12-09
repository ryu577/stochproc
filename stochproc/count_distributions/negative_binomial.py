import numpy as np
from scipy.stats import nbinom


class NegativeBinomial():
    @staticmethod
    def rvs(n,p,size=1000):
        return nbinom.rvs(n, p, size=size)



