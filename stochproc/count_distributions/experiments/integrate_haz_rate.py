import numpy as np

#########################
## Lomax
theta = .4
m=.5
n=100
t = 20

ex = m*np.log((t+theta)/theta)
print(ex)

sum1 = 0
for i in range(1000):
    ts = (np.random.uniform(size=n)**(-1/m)-1)*theta
    sum1+=sum(np.cumsum(ts)<t)
print(sum1/1000)


#########################
## Weibull
k=.3
lmb=2.0
t=20

ex = (t/lmb)**k
print(ex)

sum1 = 0
for i in range(10000):
    ts = lmb*(np.log(1/np.random.uniform(size=100)**(1/k)))
    sum1+=sum(np.cumsum(ts)<t)
sum1/10000
