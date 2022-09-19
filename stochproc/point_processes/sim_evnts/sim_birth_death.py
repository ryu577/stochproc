import numpy as np
import matplotlib.pyplot as plt
import stochproc.point_processes.sim_evnts.air_sim_renewal as asr
import argparse


def sim_brth_dth(s=50, e=65, lmb=1, mu=6, vms=5):
	n = 0; m = 0
	est1 = 0; est2 = 0; est3 = 0; est5 = 0
	for _ in range(vms):
		t = 0
		av = True
		while t < e+100:
			s1 = t
			if av:
				durtn = np.random.exponential(lmb)
			else:
				durtn = np.random.exponential(mu)
				# durtn = 0
			t += durtn
			e1 = t
			if av:
				if e1 > s and e1 < e:
					n = n+1
					est1 = est1 + durtn
					est2 = est2 + e1 - max(s, s1)
					est3 = est3 + durtn
					est5 = est5 + durtn
				elif e1 > e and s1 < e:
					m = m + 1
					est2 = est2 + e - max(s, s1)
					est3 = est3 + e - s1
					est5 = est5 + e - s1
			av = not av
	if est1 == 0:
		res1 = 0
	else:
		res1 = n/est1
	if est2 == 0: 
		res2 = 0
	else: 
		res2 = n/est2
	if est3 == 0: 
		res3 = 0
	else:
		res3 = n/est3
	if est5 == 0: 
		res5 = 0
	else:
		res5 = (n+m)/est5
	return res1, res2, res3, res5


def cmp_ests(s=100, e=120, lmb=1, mu=1, vms=20, ttx=False):
	ests1 = []; ests2 = []; ests3 = []; ests5 = []
	num_invalid_trials = 0
	for _ in range(2000):
		est_num = 0
		try:
			est1, est2, est3, est5 =\
				sim_brth_dth(s, e, lmb, mu, vms=vms)
			if ttx:
				est_num += 1 
				est1 = 1 / est1
				est_num +=1 
				est2 = 1 / est2 
				est_num +=1
				est3 = 1 / est3
				est_num +=2
				est5 = 1 / est5 
				est_num +=1
			ests1.append(est1)
			ests2.append(est2)
			ests3.append(est3)
			ests5.append(est5)
		except ZeroDivisionError: 
			num_invalid_trials += 1
			if num_invalid_trials % 100 == 0: 
				print('An estimator had no events taken into account, trial is '
					+ f'invalid. Number of invalid trials: {num_invalid_trials}')
		except:
			pass
	#plt.hist(ests1)
	#plt.show()
	return populate_res(lmb,
		                np.array(ests1),
		                np.array(ests2),
					    np.array(ests3),
					    np.array(ests5)
					   )


def cmp_ests2(s=100, e=120, lmb=15, vms=10):
	ests = []
	for _ in range(20000):
		try:
			est = asr.sim_poisson_simplified(vms=vms, s=s, e=e, lmb=lmb)
			ests.append(est)
		except:
			pass
	print(np.mean(ests))
	print(np.var(ests))


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
	# res = np.zeros((4,3))
	res = np.zeros((3,3))
	populate_arr(0, ests1, mu, res)
	populate_arr(1, ests2, mu, res)
	# populate_arr(2, ests3, mu, res)
	# populate_arr(3, ests5, mu, res)
	populate_arr(2, ests5, mu, res)
	return res


def populate_arr(ix, ests1, mu, res):
	res[ix,0] = np.mean(ests1)
	res[ix,1] = np.var(ests1)
	res[ix,2] = np.mean((ests1 - mu)**2)

if __name__=='__main__':
    # parse arguments 
    parser = argparse.ArgumentParser(description='Run simulations on TTX/event rate estimators')
    parser.add_argument('s', type=int, help='start time of window')
    parser.add_argument('e', type=int, help='end time of window')
    parser.add_argument('lmb', type=int, help='scale of exponential distribution for sampling time between incident arrivals')
    parser.add_argument('mu', type=int, help='scale of exponential distribution for sampling duration time of incidents')
    parser.add_argument('vms', type=int, help='number of VMs in simulation')
    parser.add_argument('ttx', type=bool, help='True if estimating TTX, False if estimating rate')
    args = parser.parse_args()
    s, e, lmb, mu, vms, ttx = args.s, args.e, args.lmb, args.mu, args.vms, args.ttx

    # run simulations and compare TTR estimates 
    cmp_ests(s=s, e=e, lmb=lmb, mu=mu, vms=vms, ttx=ttx)