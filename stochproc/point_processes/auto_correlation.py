import numpy as np
from scipy.stats import lomax


k=0.8
lmb=2.0

s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
n_sim=5000
for _ in range(n_sim):
    intervals = lomax.rvs(c=k, scale=(1/lmb), size=800)
    #intervals = np.random.exponential(scale=1,size=1000)
    time_stamps = np.cumsum(intervals)
    n1 = sum((time_stamps>400) * (time_stamps<900))
    n2 = sum((time_stamps>900) * (time_stamps<1400))
    s_n1n2+=n1*n2
    s_n1_sq+=n1*n1
    s_n2_sq+=n2*n2
    s_n1+=n1
    s_n2+=n2

cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
corln = cov/np.sqrt(v_n1*v_n2)

print("correlation: " +str(corln))

#####

k=0.8
lmb=2.0

s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
n_sim=5000
for _ in range(n_sim):
    intervals1 = lomax.rvs(c=k, scale=(1/lmb), size=800)
    intervals2 = lomax.rvs(c=k, scale=(1/lmb), size=800)
    #intervals = np.random.exponential(scale=1,size=1000)
    time_stamps1 = np.cumsum(intervals1)
    time_stamps2 = np.cumsum(intervals2)
    n1 = sum((time_stamps1>400) * (time_stamps1<900))
    n2 = sum((time_stamps2>400) * (time_stamps2<900))
    s_n1n2+=n1*n2
    s_n1_sq+=n1*n1
    s_n2_sq+=n2*n2
    s_n1+=n1
    s_n2+=n2

cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
corln = cov/np.sqrt(v_n1*v_n2)

print("correlation: " +str(corln))


#####

k=7.0
lmb=2.0## Notice we doubled the scale so the mean is halved.

s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
n_sim=5000
for _ in range(n_sim):
    intervals = lomax.rvs(c=k, scale=(1/lmb), size=2000)
    #intervals = np.random.exponential(scale=1,size=1000)
    time_stamps = np.cumsum(intervals)
    bi_furcator = np.random.choice(2,size=len(time_stamps))
    time_stamps1 = time_stamps[bi_furcator==1]
    time_stamps2 = time_stamps[bi_furcator==0]
    n1 = sum((time_stamps1>50) * (time_stamps1<90))
    n2 = sum((time_stamps2>50) * (time_stamps2<90))
    s_n1n2+=n1*n2
    s_n1_sq+=n1*n1
    s_n2_sq+=n2*n2
    s_n1+=n1
    s_n2+=n2

cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
corln = cov/np.sqrt(v_n1*v_n2)

print("correlation: " +str(corln))

#####
## Check mean

k=1.2
lmb=1.0

s_n1=0
n_sim=5000
for _ in range(n_sim):
    intervals = lomax.rvs(c=k, scale=(1/lmb), size=1200)
    #intervals = np.random.exponential(scale=1,size=1000)
    time_stamps = np.cumsum(intervals)
    #n1 = sum((time_stamps>100) * (time_stamps<200))
    n1 = sum(time_stamps<100)
    s_n1+=n1

e_n1 = s_n1/n_sim

print("simulated mean: " +str(e_n1))
print("actual mean-1: " +str(k*200*lmb))
print("actual mean-2: " +str((k-1)*200*lmb))

#####
## Poisson mixture..

k=1.2
theta=1.0

s_n1=0
n_sim=5000
for _ in range(n_sim):
    t=0
    n=0
    while t<100:
        lm = np.random.gamma(k,theta)
        #lm=1.2
        t+=np.random.exponential(1/lm)
        n+=(t<100)#*(t>100)
    s_n1+=n

e_n1 = s_n1/n_sim

print("Simulated mean: " + str(e_n1))
print("Theoretical mean:" + str(k*100))
