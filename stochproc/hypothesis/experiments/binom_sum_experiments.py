import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.special import hyp2f1

def binom_partial_sum(n,p=.5):
    b_sum=0
    for j in range(int(n/2)+1):
        b_sum+=comb(n,j)*(1+p)**j
    return b_sum#/(2+p)**n

def partial_sum_closed_form(n,p=0.5):
    return 1-comb(n,n/2+1)*hyp2f1(1,1-n/2, n/2+2, -p-1)*(1+p)**(n/2+1)/(2+p)**n


sums = np.array([binom_partial_sum(i,p=0.1) for i in range(11,501,2)])
sums2 = ((2+0.1)*0.9999)**np.arange(11,501,2)/(2+0.1)**np.arange(11,501,2)

plt.plot(np.arange(11,501,2),sums,label="partial sum")
plt.plot(np.arange(11,501,2),sums2,label="approx")
plt.legend()
plt.show()

#https://www.wolframalpha.com/input/?i=sum+%28n+choose+j%29+%281%2Bp%29%5Ej%2C+j%3Dn%2F2%2B1+to+n
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839521/
