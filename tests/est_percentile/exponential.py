import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


def prcntl_bias(q, n, interpolate=1):
	lt = int(np.floor(q*(n-1)))
	summ = 0
	for ix in range(lt+1):
		summ += 1/(n-ix)
	if interpolate==1:
		summ += np.modf(q*(n-1))[0]/(n-lt-1)
	elif interpolate==2:
		summ += expon_frac(q, n)/(n-lt-1)
	return -np.log(1-q)-summ


def analyze_bias(n=55, save_close=True):
	biases1 = [prcntl_bias(q,n) for q in np.arange(0.1,1.0,0.01)]
	biases2 = [prcntl_bias(q,n,0) for q in np.arange(0.1,1.0,0.01)]
	biases3 = [prcntl_bias(q,n,2) for q in np.arange(0.1,1.0,0.01)]
	plt.plot(np.arange(0.1,1.0,0.01), biases1)
	plt.plot(np.arange(0.1,1.0,0.01), biases2)
	plt.plot(np.arange(0.1,1.0,0.01), biases3)
	plt.axhline(0, color="black")
	fn1 = lambda q: prcntl_bias(q,n)
	rt = root_scalar(fn1,bracket=[0,1],method='bisect').root
	print("Unbiased percentile: " + str(rt))
	plt.axvline(rt,color="black")
	plt.title("Unbiased percentile for n=" + str(n) + " is: " + str(rt))
	if save_close:
		plt.savefig('plots/sample_' + str(n) + '.png')
		plt.close()


def expon_frac(q, n):
	lt = int(np.floor(q*(n-1)))
	summ = 0
	for ix in range(lt+1):
		summ += 1/(n-ix)
	return (-np.log(1-q)-summ)*(n-lt-1)


for n in range(15,1005,10):
	analyze_bias(n)


