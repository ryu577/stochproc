import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def binom_partial_sum(n,p=.5):
    b_sum=0
    for j in range(int(np.ceil(n/2))+1):
        b_sum+=comb(n,j)*(1+p)**j
    return b_sum**(1/n)

sums = np.array([binom_partial_sum(i,p=3.0) for i in range(3,501,1)])

plt.plot(np.arange(3,501,1),sums,label="partial sum")
plt.legend()
plt.show()

#https://www.wolframalpha.com/input/?i=sum+%28n+choose+j%29+%281%2Bp%29%5Ej%2C+j%3Dn%2F2%2B1+to+n
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839521/
