import numpy as np
from scipy import stats
from scipy.stats import poisson


cnt = 0; n=1e4; alpha=0.05
for sim in range(int(n)):
    lmb=12.0; mu=12.0; t=10e3/4/100; s=10e3/4/100
    pois1 = poisson.rvs(lmb*t,size=20)/t
    pois2 = poisson.rvs(mu*s,size=20)/s
    mean1 = np.mean(pois1); std1=np.std(pois1)
    mean2 = np.mean(pois2); std2=np.std(pois2)
    if mean2>mean1:
        t_score = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=20, \
                               mean2=mean2, std2=std1, nobs2=20, \
                               equal_var=False)
        if t_score.pvalue/2 < alpha:
            cnt+=1

print(cnt/n)

