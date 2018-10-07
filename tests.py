import numpy as np
from stochproc.competitivecointoss.smallmarkov import *


def tst_eigen_coefs():
	coef = np.array(np.linalg.inv(np.linalg.eig(a)[1]).T[2])[0]*np.array(np.linalg.eig(a)[1][0])[0]
	#sum(coef*np.linalg.eig(a)[0]**5) == np.dot(a,np.dot(a,np.dot(a,np.dot(a,a))))[0,2]
	a = np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]])
	a_c, a_e = get_coefs(a)
	return coef[0] = a_c[0]

def tst_diagonalizable():
	a = np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]])
	## Get back A from its eigen decomposition.
	a1 = np.dot(np.dot(np.linalg.eig(a)[1],np.diag(np.linalg.eig(a)[0])),np.linalg.inv(np.linalg.eig(a)[1]))
	return a[0,0] == a1[0,0]


def tst_powers1():
	powrs = get_n_powers([.5,.5,1])
	return powrs[1] == 1

def tst_powers2():
	get_n_powers([.5,.5,1,.4,.4,.4,1])
	return powrs[5] == 1


