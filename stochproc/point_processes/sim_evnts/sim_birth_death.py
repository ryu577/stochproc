import numpy as np
import matplotlib.pyplot as plt
import stochproc.point_processes.sim_evnts.air_sim_renewal as asr


def sim_brth_dth(s=50, e=65, lmb=1, mu=6, vms=5):
	n = 0; m = 0
	est1 = 1e-30; est2 = 1e-30; est3 = 1e-30; est5 = 1e-30
	for i in range(vms):
		t = 0
		av = True
		while t < e+500:
			s1 = t
			if av:
				durtn = np.random.exponential(lmb)
			else:
				durtn = np.random.exponential(mu)
			t += durtn
			e1 = t
			if av:
				if e1 > s and e1 < e:
					n = n + 1
					est1 = est1 + durtn
					est2 = est2 + e1 - max(s, s1)
					est3 = est3 + durtn
					est5 = est5 + durtn
				elif e1 >= e and s1 < e:
					m = m + 1
					est2 = est2 + e - max(s, s1)
					est3 = est3 + e - s1
					est5 = est5 + e - s1
			av = not av
	res1 = n/est1
	res2 = n/est2
	res3 = n/est3
	res5 = (n+m)/est5
	return res1, res2, res3, res5


def cmp_ests(s=100, e=120, lmb=1, mu=1, vms=20):
	ests1 = []; ests2 = []; ests3 = []; ests5 = []
	for i in range(1000):
		try:
			est1, est2, est3, est5 =\
				sim_brth_dth(s, e, lmb, mu, vms=vms)
			ests1.append(est1)
			ests2.append(est2)
			ests3.append(est3)
			ests5.append(est5)
		except:
			pass
	plt.hist(ests2)
	plt.axvline(1/lmb, color="black")
	plt.show()
	return populate_res(lmb,
		                np.array(ests1),
		                np.array(ests2),
					    np.array(ests3),
					    np.array(ests5)
					   )


def cmp_ests2(s=100, e=120, lmb=15, vms=10):
	ests = []
	means = []
	for _ in range(10000):
		try:
			est, e_mean = asr.sim_poisson_simplified(vms=vms, s=s, e=e, lmb=lmb)
			ests.append(est)
			if e_mean is not None:
				means.append(e_mean)
		except:
			pass
	plt.hist(means)
	plt.axvline(lmb, color="black")
	plt.show()
	print(np.mean(ests))
	print(np.var(ests))
	print("###############")
	print(np.mean(means))
	print(np.var(means))


def populate_res(mu, ests1, ests2, ests3, ests5):
	print(np.mean(ests1))
	print(np.var(ests1))
	print("######")
	print(np.mean(ests2))
	print(np.var(ests2))
	print("######")
	print(np.mean(ests3))
	print(np.var(ests3))
	print("######")
	print(np.mean(ests5))
	print(np.var(ests5))
	#plt.hist(ests1)
	res = np.zeros((4,3))
	populate_arr(0, ests1, mu, res)
	populate_arr(1, ests2, mu, res)
	populate_arr(2, ests3, mu, res)
	populate_arr(3, ests5, mu, res)
	return res


def populate_arr(ix, ests1, mu, res):
	res[ix,0] = np.mean(ests1)
	res[ix,1] = np.var(ests1)
	res[ix,2] = np.mean((ests1 - mu)**2)

