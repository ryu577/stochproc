import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def single_machine(t, lmb=1/10, mu=1/3):
	"""
	Simulates a single machine. The mean time to failure (MTBF)
	for the machine is a exponential with rate lambda. The mean
	time to repair once the machine goes down is another exponential
	with mean mu. 
	"""
	running_time = 0
	while True:
		up = True
		up_time = np.random.exponential(1/lmb)
		running_time += up_time
		if running_time > t:
			break
		up = False
		down_time = np.random.exponential(1/mu)
		running_time += down_time
		if running_time > t:
			break
	return up


def updown(t):
	ups = 0
	ups_arr = []
	for j in range(1):
		for i in range(1000):
			upp = single_machine(t)
			ups += upp
		ups_arr.append(upp/1000)
	return ups/1000, np.std(ups_arr)


def closed_form(t,lmb=1/10,mu=1/3):
	return mu/(mu+lmb)+lmb/(lmb+mu)*np.exp(-(lmb+mu)*t)


def plot_one_mc_sim():
	for i in range(10):
		probs = []
		stds = []
		for t in range(1,100):
			prob, std = updown(t)
			probs.append(prob)
			stds.append(std)
		plt.plot(np.arange(1,100), probs,alpha=0.4,color='pink')

	xs = np.arange(1,100)
	plt.plot(xs, closed_form(xs),color='red')
	plt.xlabel('Time')
	plt.ylabel('Reliability of system')
	plt.show()


def k_of_n_sys(p, k=2, n=3):
	res = 0
	for l in range(k,n+1):
		res += comb(n,l)*p**l*(1-p)**(n-l)
	return res

def two_of_three_with_network(p, s):
	return 3*p**2*(1-p)*s + p**3*(1-(1-s)**3)


def k_of_n_w_network(p, s, k=2, n=3):
	res = 0
	for l in range(k,n+1):
		res += comb(n,l)*p**l*(1-p)**(n-l)*is_master_available(s,k,l,0.5)
	return res


def is_master_available(s, k=2, l=3, s1=None):
	res = 0
	if s1 is None:
		s1 = s
	for sim in range(30000):
		arr = np.zeros(l)
		succ = 0
		for i in range(l):
			for j in range(i+1,l):
				if np.random.uniform() < s1:
					succ += 1
					arr[i] += 1
					arr[j] += 1
		if max(arr) < k-1:
			res += s**succ*(1-s)**(l*(l-1)/2-succ)/s1**succ/(1-s1)**(l*(l-1)/2-succ)
	return 1-res/30000


def two_of_three_reliability_mc(t=10, mu=0.7, lmb=9.3):
	sys_work = 0	
	for i in range(1000):
		summ = single_machine(t,lmb,mu)+single_machine(t,lmb,mu)+single_machine(t,lmb,mu)
		if summ >= 2:
			sys_work += 1
	return sys_work/1000


def two_of_three_reliability(t=10, mu=0.7, lmb=9.3):
	p = closed_form(t, lmb, mu)
	return k_of_n_sys(p,2,3)


def two_of_three_reliability_mc_importance(t=10, mu=0.7, lmb=9.3):
	sys_work = 0	
	p = closed_form(t,lmb,mu)
	q = closed_form(t,lmb,mu*5)
	for i in range(1000):
		summ = single_machine(t,lmb,mu*5)+single_machine(t,lmb,mu*5)+single_machine(t,lmb,mu*5)
		if summ == 2:
			sys_work += (p**2*(1-p))/(q**2*(1-q))
		elif summ == 3:
			sys_work += p**3/q**3
	return sys_work/1000


