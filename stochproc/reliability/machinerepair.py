import numpy as np
import matplotlib.pyplot as plt


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



