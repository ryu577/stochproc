import numpy as np
from scipy.stats import lomax

k=0.8
lmb=2.0

s_n1n2=0; s_n1=0; s_n2=0; s_n1_sq=0; s_n2_sq=0
n_sim=1000
for _ in range(n_sim):
    intervals = lomax.rvs(c=k, scale=(1/lmb), size=800)
    #intervals = np.random.exponential(scale=1,size=1000)
    time_stamps = np.cumsum(intervals)

    n1 = sum((time_stamps>300) * (time_stamps<350))
    n2 = sum((time_stamps>350) * (time_stamps<400))
    s_n1n2+=n1*n2
    s_n1_sq+=n1*n1
    s_n2_sq+=n2*n2
    s_n1+=n1
    s_n2+=n2

cov = s_n1n2/n_sim-(s_n1/n_sim)*(s_n2/n_sim)
v_n1=s_n1_sq/n_sim-(s_n1/n_sim)**2
v_n2=s_n2_sq/n_sim-(s_n2/n_sim)**2
corln = cov/np.sqrt(v_n1*v_n2)


