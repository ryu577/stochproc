import numpy as np
from distributions.lomax import Lomax
from distributions.weibull import Weibull
from scipy.stats import poisson
import matplotlib.pyplot as plt


## Over dispersed
k = 1.2; lmb=25; durtn = 1.0
mean = Lomax.mean_s(k,lmb)
aa=np.array([sum(np.cumsum(Lomax.samples_(k,lmb,size=100))<durtn) for _ in range(1000)])

expctd_mean = durtn/mean
actual_mean = np.mean(aa)
print("Diff:" + str(actual_mean - expctd_mean))
var = np.var(aa)


## Under dispersed
# k=0.4; lmb=5.0; durtn=20.0
def weibull_to_count(k=0.4,lmb=1.0,durtn=20.0):
    mean = Weibull.mean_s(k,lmb)
    aa1=np.array([sum(np.cumsum(Weibull.samples_(k,lmb,size=100))<durtn) for _ in range(1000)])
    expctd_mean = durtn/mean
    actual_mean = np.mean(aa1)
    print("Diff:" + str(actual_mean - expctd_mean))
    #var = np.var(aa1)


###############################################
#### Some experiments..

## What does increasing lmb do to Weibull?
xs = np.array(np.arange(1.0,5.0,.1))
ys = Weibull().pdf(x=xs,k=0.5,lmb=10.0)
plt.plot(xs,ys,label='lmb:100.0')

ys = Weibull().pdf(x=xs,k=0.5,lmb=50.0)
plt.plot(xs,ys,label='lmb:50.0')

ys = Weibull().pdf(x=xs,k=0.5,lmb=100.0)
plt.plot(xs,ys,label='lmb:10.0')

plt.legend()
plt.show()

## Conclusion: Increasing lmb spreads x out more and so, 
# makes the distribution more like exponential 
# (flattens hazard rate).

from scipy.stats import binom, nbinom
from scipy.special import binom as bn
import matplotlib.pyplot as plt

def fn(alp, n=10):
    return binom.cdf(binom.isf(alp,n,.5),n,.5+.1)

alps = np.arange(0,1,0.01)
plt.plot(alps,fn(alps,100))
plt.plot(alps,fn(alps,200))
plt.plot(alps,fn(alps,300))
plt.show()


fig, axs = plt.subplots(2)
ns = np.arange(100)
#axs[0].plot(ns,binom.pmf(ns,100,.5))
#axs[1].plot(ns,binom.pmf(ns,100,.5+.1))
axs[0].plot(ns,binom.sf(ns,100,.5))
axs[1].plot(ns,binom.cdf(ns,100,.5+.1))

plt.show()

#####
#
def mixed_conditional(j,n=100,m=5):
    return (2*m-1)/(n+2*m-1)*bn(n,j)*bn(2*m-2,m-1)/bn(n+2*m-2,j+m-1)

sum(mixed_conditional(np.arange(101)))

plt.plot(np.arange(101),mixed_conditional(np.arange(101)))

######
#
nbinom.pmf(2,10,.5)

def nbinom_sum(n,m1,p1,m2,p2):
    summ = 0
    for i in range(n+1):
        summ += nbinom.pmf(i,m1,p1)*nbinom.pmf(n-i,m2,p2)
    return summ


def conditional_prob(j,n=50,m1=10,p1=.5,m2=10,p2=.6):
    return nbinom.pmf(j,m1,p1)*nbinom.pmf(n-j,m2,p2)/nbinom_sum(n,m1,p1,m2,p2)

sum(conditional_prob(np.arange(51)))

plt.plot(np.arange(51),conditional_prob(np.arange(51)))
plt.show()

