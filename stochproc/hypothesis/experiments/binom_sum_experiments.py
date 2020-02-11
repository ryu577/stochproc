import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, hyp2f1, erf


def binom_partial_sum(n,p=.5):
    b_sum=0
    for j in range(int(np.ceil(n/2))+1):
        b_sum+=comb(n,j)*(1+p)**j
    return b_sum**(1/n)

def binom_approx(m,p=.5):
	alpha=1+p
	n=m/2
	trm = np.exp(n*np.log(alpha)**2/4)/2*\
			( erf( (n*np.log(alpha)+2*n+1)/2/np.sqrt(n) )\
			- erf( (n*np.log(alpha)-1)/2/np.sqrt(n)) )
	return trm**(1/m)*2*np.sqrt(alpha)


sums = np.array([binom_partial_sum(i,p=0.7) for i in range(3,501,1)])
sums2 = np.array([binom_approx(i,p=0.7) for i in range(2,500,2)])
plt.plot(np.arange(3,501,1),sums,label="partial sum")
plt.plot(np.arange(2,500,2),sums2,label="partial sum approx")


def partial_sum_closed_form(n,p=0.5):
    return 1-comb(n,n/2+1)*hyp2f1(1,1-n/2, n/2+2, -p-1)*(1+p)**(n/2+1)/(2+p)**n


sums = np.array([binom_partial_sum(i,p=0.5) for i in range(11,501,2)])
sums2 = (2+0.5-0.5**0.5)**np.arange(11,501,2)/(2+0.5)**np.arange(11,501,2)

plt.plot(np.arange(11,501,2),sums,label="partial sum")
plt.plot(np.arange(11,501,2),sums2,label="approx")
plt.legend()
plt.show()


#https://www.wolframalpha.com/input/?i=sum+%28n+choose+j%29+%281%2Bp%29%5Ej%2C+j%3Dn%2F2%2B1+to+n
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5839521/

