import numpy as np
from scipy import stats
from scipy.stats import poisson
from stochproc.hypothesis.rate import *


def compare_tests(n=1e4, alpha=0.05, lmb=12.0, mu=12.0):
    cnt=0; cnt1=0
    for _ in range(int(n)):
        t=10e3/4/100; s=10e3/4/100
        n1s = poisson.rvs(lmb*t,size=20)
        n2s = poisson.rvs(mu*s,size=20)
        rate1 = n1s/t
        rate2 = n2s/s
        n1 = sum(n1s); n2 = sum(n2s)
        d = n2/(20*s)-n1/(20*t)
        lmb_mix = (n1+n2)/(t+s)/20
        p_val2 = pois_diff_sf(d,lmb_mix,lmb_mix,t,s)
        if p_val2 < alpha and n2/s>n1/t:
            cnt1+=1
        mean1 = np.mean(rate1); std1=np.std(rate1)
        mean2 = np.mean(rate2); std2=np.std(rate2)
        if mean2>mean1:
            t_score = stats.ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=20, \
                                mean2=mean2, std2=std2, nobs2=20, \
                                equal_var=False)
            if t_score.pvalue/2 < alpha:
                cnt+=1
    print(cnt/n)
    print(cnt1/n)


