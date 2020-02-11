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


