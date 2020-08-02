import numpy as np
import pandas as pd

def est_moments(eps=0.3,lmb=7.0, t=1.0, n_sim=1000):
    lmbs = []
    for _ in range(n_sim):
        u=np.random.uniform(eps,1)
        sum_t=np.random.exponential(1/lmb); cnt=0
        while sum_t<u:
            cnt+=1
            sum_t+=np.random.exponential(1/lmb)
        lmbs.append(cnt/u)
    return np.mean(lmbs), np.var(lmbs)

eps=0.318; lmb=7.0
var1 = -lmb*np.log(eps)/(1-eps)
mu_sim, var_sim = est_moments(eps,lmb,n_sim=500000)

print("Actual mean:" + str(lmb))
print("Simulated mean:" + str(mu_sim))

print("Actual variance:" + str(var1))
print("Simulated variance:" + str(var_sim))

###############
##

up=True
mu=1.0;
sig=2.0;
t=1000;
ts=[]; cum_t=0; states=[];
starts=[]; ends=[]
while cum_t<t:
    starts.append(cum_t)
    rate = mu if up else sig
    state = "up" if up else "down"
    t1 = np.random.exponential(1/rate)
    ts.append(t1)
    states.append(state)
    cum_t += t1
    up=not up
    ends.append(cum_t)

dat = pd.DataFrame({"durtn":ts,"state":states,"start":starts,"end":ends})

###################
##Availablity along with turning VMs off.
up=True
turn_on=1.0
turn_off=2.0
lmb = 3.0
mu=10
t=10000
ts=[]; cum_t=0; states=[]
starts=[]; ends=[]
while cum_t<t:
    starts.append(cum_t)
    rate = mu if up else sig
    state = "up" if up else "down"
    t1 = np.random.exponential(1/rate)
    ts.append(t1)
    states.append(state)
    cum_t += t1
    up=not up
    ends.append(cum_t)

dat = pd.DataFrame({"durtn":ts,"state":states,"start":starts,"end":ends})


##########################
#### Storage availability.
availab = 0
nsim=10000
lmb=10.0
x=2.0
t=1000    
for i in range(nsim):
    up=True
    cum_t=0
    while cum_t<t:
        state = "up" if up else "down"
        t1 = np.random.exponential(1/lmb) if up else x
        cum_t += t1
        up=not up        
    if state == "up":
        availab+=1

print("Simulated availability:"+str(availab/nsim))
print("Formula 2 availability:"+str(1/(1+lmb*x)))
print("Formula 1 availability:"+str(np.exp(-lmb*x)))

