import numpy as np

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

