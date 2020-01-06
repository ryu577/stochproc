import numpy as np
from stochproc.competitivecointoss.smallmarkov import *
from stochproc.reliability.machinerepair import *
from stochproc.hypothesis.rate_test import *
import stochproc.hypothesis.rate_test as xtst
from stochproc.hypothesis.hypoth_tst_simulator import *
from stochproc.hypothesis.binom_test import *
from stochproc.count_distributions.compound_poisson import CompoundPoisson
from stochproc.count_distributions.negative_binomial import rvs_mxd_poisson
import matplotlib.pyplot as plt
import pytest
from importlib import reload


def tst_eigen_coefs():
	coef = np.array(np.linalg.inv(np.linalg.eig(a)[1]).T[2])[0]*np.array(np.linalg.eig(a)[1][0])[0]
	#sum(coef*np.linalg.eig(a)[0]**5) == np.dot(a,np.dot(a,np.dot(a,np.dot(a,a))))[0,2]
	a = np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]])
	a_c, a_e = get_coefs(a)
	assert coef[0] == a_c[0]

def tst_diagonalizable():
	a = np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]])
	## Get back A from its eigen decomposition.
	a1 = np.dot(np.dot(np.linalg.eig(a)[1],np.diag(np.linalg.eig(a)[0])),np.linalg.inv(np.linalg.eig(a)[1]))
	assert a[0,0] == a1[0,0]

def tst_powers1():
	powrs = get_n_powers([.5,.5,1])
	assert powrs[1] == 1

def tst_powers2():
	powrs = get_n_powers([.5,.5,1,.4,.4,.4,1])
	assert powrs[5] == 1

def tst_power_series():
	assert abs(sum_inf_n_powklambda_pown(.25,3) - .25/.75**4*(2+.25**2))<1e-13

def tst_k_of_n_netwrk():
	assert abs(is_master_available(0.97,2,3,0.5)-(1-(1-0.97)**3)) < 1e-4

def tst_three_of_four_sim():
	assert abs(three_of_four_connectivity(0.5)-is_master_available(0.5,3,4,sure_conncn={},nsim=1000000))<1e-3

def tst_winning_at_nth_toss():
	start2 = np.array([1,0,0,0])
	m_4 = np.matrix([[.5,.5,0,0], [.5,0,.5,0],[.5,0,0,.5], [0,0,0,1]])
	q_n = np.array([np.dot(start2, np.linalg.matrix_power(m_4,n))[0,3]\
                              for n in range(100)])
	q_n_minus_1 = np.array([np.dot(start2, np.linalg.matrix_power(m_4,n))[0,2]\
                              for n in range(100)])
	assert sum(np.diff(q_n)[:20] - q_n_minus_1[:20]/2) == 0

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


