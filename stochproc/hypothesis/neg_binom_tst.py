import numpy as np
from scipy.stats import nbinom
from stochproc.count_distributions.negative_binomial import NegativeBinomial
from scipy.stats import norm


r0=1.4; mu=0.7; rr=1.15; r1=r0*rr
t=mu/r0; k=0.4
n0=1035; n1=1035

nb_null = NegativeBinomial(mu=mu,k=k,t=t)
nb_alt = NegativeBinomial(mu=mu*rr,k=k,t=t)

var_null = 4/mu/(r0+r1)+2*k
st_dev_null = np.sqrt(var_null)

type_2_errs = 0
for _ in range(1000):
    x_s = 0
    for i in range(n0):
        x_i = nb_null.rvs()
        x_s += x_i

    y_s = 0
    for j in range(n1):
        y_j = nb_alt.rvs()
        y_s += y_j

    r0_est = x_s/n0/t; r1_est = y_s/n1/t
    z_stat = (np.log(r1_est)-np.log(r0_est))/st_dev_null
    p_val = norm.sf(z_stat)
    type_2_errs += p_val>0.05

print(type_2_errs/1000)

