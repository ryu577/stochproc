import numpy as np
from scipy.stats import binom
from stochproc.hypothesis.rate_test import *
import stochproc.hypothesis.rate_test as xtst
from stochproc.hypothesis.hypoth_tst_simulator import *
from stochproc.hypothesis.binom_test import *
from stochproc.count_distributions.compound_poisson import CompoundPoisson
from stochproc.count_distributions.negative_binomial import rvs_mxd_poisson
import matplotlib.pyplot as plt
import pytest
from importlib import reload

## Tests for hypothesis testing and beta calculations.

def tst_binom_tst():
	beta_1 = binom_tst_beta(p_null=0.5,p_alt=0.6,n=10,alpha_hat=0.05)
	beta_2 = binom_tst_beta_sim(p_null=0.5,p_alt=0.6,n=10,alpha_hat=0.05,n_sim=50000)
	assert abs(beta_1-beta_2)/beta_1<1e-2


def tst_ump_poisson_on_poisson():
	beta_1 = UMPPoisson.beta_on_poisson_closed_form(t1=25,t2=25,\
	                lmb_base=12,effect=3,alpha=0.5)

	dist_rvs_poisson = lambda lmb,t: poisson.rvs(lmb*t)
	alphas, betas, alpha_hats = alpha_beta_curve(dist_rvs_poisson,n_sim=10000, lmb=12, t1=25, t2=25, \
	                        scale=1.0, dellmb = 3.0)

	ix=np.argmin((alphas-0.5)**2)
	print(alphas[ix])
	beta_2 = betas[ix]

	beta_3 = UMPPoisson.beta_on_poisson_closed_form3(t1=25,t2=25,\
	                lmb_base=12,effect=3)
	assert abs(beta_2-beta_1)/beta_2<1e-3 and abs(beta_3-beta_1)/beta_3<1e-3


##Now for compound poisson.
def tst_ump_poisson_on_comp_poisson():
	## Only works for l=1!	
	l=1; p=.3
	beta_1 = UMPPoisson.beta_on_poisson_closed_form(t1=25,t2=25,\
	                lmb_base=12*l*p,effect=1*l*p,alpha=0.1)
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,l,p,compound='binom')
	alphas, betas, alpha_hats = alpha_beta_curve(dist_rvs_compound,n_sim=10000, lmb=12, t1=25, t2=25, \
	                        scale=1.0, dellmb = 1)
	ix=np.argmin((alpha_hats-0.1)**2)
	print(alpha_hats[ix])
	beta_2 = betas[ix]
	assert (beta_2-beta_1)/beta_1 < 1e-3


##Now mixed Poisson (negative binomial).
def tst_ump_poisson_on_neg_binom():
	beta_1 = UMPPoisson.beta_on_negbinom_closed_form(t1=10,t2=10,\
	                theta_base=5,m=100.0,deltheta=1,alpha=0.1,cut_dat=1e4)[0]

	beta_1_50 = UMPPoisson.beta_on_negbinom_closed_form(t1=10,t2=10,\
	                theta_base=5,m=100.0,deltheta=1,alpha=0.5,cut_dat=1e4)[0]

	rvs_mxd_poisson_0 = lambda t: rvs_mxd_poisson(t,5,100)
	rvs_mxd_poisson_1 = lambda t: rvs_mxd_poisson(t,4,100)

	alphas2,betas2,alpha_hats2 = alpha_beta_tracer(rvs_mxd_poisson_0, rvs_mxd_poisson_1,t1=10,t2=10)
	ix=np.argmin((alpha_hats2-0.1)**2)
	print(alpha_hats2[ix])
	beta_2 = betas2[ix]
	print(beta_2)

	beta_3 = UMPPoisson.beta_on_negbinom_closed_form3(t1=10,t2=10,\
						theta_base=5,m=100.0,deltheta=1)
	assert abs(beta_1-beta_2)/beta_1<1e-3 and abs(beta_1_50-beta_3)/beta_1_50<1e-9



##If you want to be able to reload in interactive console without restarting, 
# use rtst.<method>
# instead of the method name directly.

def tst_ump_poisson_on_determinist_cmpnd_alpha(plot=False):
	alphas,alpha_hats,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
											lmb=10.0,t1=10,t2=10,l=3)
	ix=np.argmin((alpha_hats-0.05)**2)
	print(alpha_hats[ix])
	alpha = alphas[ix]
	l=3; p=1.0
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,l,p,compound='binom')
	alphas1, betas1, alpha_hats1 = alpha_beta_curve(dist_rvs_compound,n_sim=10000, \
								lmb=10, t1=10, t2=10, scale=1.0, dellmb = 0)
	ix=np.argmin((alpha_hats1-0.05)**2)
	print(alpha_hats1[ix])
	alpha1 = alphas1[ix]
	if plot:
		plt.plot(alpha_hats,alphas,label='Closed form')
		plt.plot(alpha_hats1,alphas1,label='Simulated')
		plt.legend()
		plt.show()
	assert abs(alpha1-alpha)/alpha1 < 1e-3


def weird_behavior():
	UMPPoisson.beta_on_poisson_closed_form(t1=0.5,t2=0.1,\
                                lmb_base=2,\
                                alpha=0.1,effect=10.0)
	#Out[19]: (0.7202482866175243, 0.9999999865720979)
	UMPPoisson.beta_on_poisson_closed_form(t1=0.6,t2=0.1,\
                                lmb_base=2,\
                                alpha=0.1,effect=10.0)
	#Out[19]: (0.7359201224192129, 0.9999999938916904)
	## Why does beta incrase as we increase t1??


##Same effect from increasing t2:
def weird_behavior2():
	UMPPoisson.beta_on_poisson_closed_form(t1=0.5,t2=0.23,\
									lmb_base=2,\
									alpha=0.1,effect=10.0)
	#Out[22]: (0.46287986945821635, 0.9999999800514148)

	UMPPoisson.beta_on_poisson_closed_form(t1=0.5,t2=0.3,\
									lmb_base=2,\
									alpha=0.1,effect=10.0)
	#Out[23]: (0.500373244503035, 0.9999999794679547)


def beta_with_t():
	betas=[]
	ts = np.arange(1.5,6.5,0.1)
	for t in ts:
		beta = UMPPoisson.beta_on_poisson_closed_form(\
									t1=t,t2=t,\
									lmb_base=10,\
									alpha=0.1,effect=5.0)
		betas.append(beta[0])
	plt.xlabel('VM centuries in both groups')
	plt.ylabel('False negative rate for the test holding false positive at 15%')
	plt.plot(ts,betas)
	plt.show()


def concrete_case_rising_beta():
	UMPPoisson.beta_on_poisson(\
								t1=0.695,t2=0.23,\
								lmb_base=2,\
								alpha=0.1,effect=10.0)
	UMPPoisson.beta_on_poisson(\
								t1=0.4988,t2=0.23,\
								lmb_base=2,\
								alpha=0.1,effect=10.0)
	alpha_hats = np.arange(0.001,1.0,0.0001)
	alphas = binom.sf(binom.isf(alpha_hats,10,0.5),10,0.5)
	plt.plot(alpha_hats,alphas)
	x1, y1 = [0, 1], [0, 1]
	x2, y2 = [0, 1], [0, 1]
	plt.plot(x1, y1, marker = 'o')
	plt.show()



