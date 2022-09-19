import numpy as np
import matplotlib.pyplot as plt
import argparse

def est_mean(s, e, lmb=1, mu=1):
	'''
    Performs a simulation of incidents arriving from time t=0 to time 
    e + 100. Computes estimators for TTR (time to resolution)

    Parameters: 
    s (int) - start time of time window 
    e (int) - end time of time window 
    lmb (int) - scale of exponential distribution for sampling time 
    between incident arrivals
    mu (int) - scale of exponential distribution for sampling duration
    time of incidents 
    
    Returns: 
    tuple of estimators - tuple[int]
    (est1, est2, est5)

    est1 (float) - second E(TTR) estimator 
    est1 = sum(e_i - s_i) / n1 where s < e_i < e and n1 = number of incidents
    where the constraint is met 

    est2 (float) - third E(TTR) estimator 
    est2 = sum(d_i) / n2 where n2 = number of incidents where s_i < e < e
    d_i = {
        e_i - max(s, s_i), when s < e_i < e
        e - max(s, s_i), when s_i < e < e_i 
    }

    est5 (float) - sixth E(TTR) estimator 
    est5 = (est5_1 + est5_2) / (n5_1 + n5_2)
    est5_1 = sum(e_i - s_i) where s < e_i < e and n5_1 = number of 
    incidents where the constraint is met 
    est5_2 = sum(e - s_i) where s_i < e < e_i and n5_2 = number of incidents
    where the constriant is met 

    '''
	t = 0
	est1 = 0
	est2 = 0
	est5 = 0
	n = 0
	m = 0
	while t < e+100:
		t_del = np.random.exponential(lmb)
		t = t + t_del
		s1 = t
		durtn = np.random.exponential(mu)
		e1 = t + durtn
		if e1 > s and e1 < e:
			n = n + 1
			est1 = est1 + durtn
			est2 = est2 + e1 - max(s, s1)
			est5 = est5 + durtn
		elif e1 > e and s1 < e:
			m = m + 1
			est2 = est2 + e - max(s, s1)
			est5 = est5 + e - s1
	res1 = est1/n
	res2 = est2/n
	res5 = est5/(n+m)
	return res1, res2, res5


# Tests..
# est3 >= est1.
def cmp_ests(s=100, e=120, lmb=1, mu=1, ttx=True):
	ests1 = []; ests2 = []; ests5 = []
	num_invalid_trials = 0
	for _ in range(2000):
		try:
			est1, est2, est5 = est_mean(s, e, lmb, mu)
			if not ttx: 
				if est1 != 0: 
					est1 = 1 / est1
				if est2 != 0: 
					est2 = 1 / est2
				if est5 != 0: 
					est5 = 1 / est5
			ests1.append(est1)
			ests2.append(est2)
			ests5.append(est5)
		except ZeroDivisionError: 
			num_invalid_trials += 1
			if num_invalid_trials % 100 == 0: 
				print('An estimator had no events taken into account, trial is '
					+ f'invalid. Number of invalid trials: {num_invalid_trials}')
		except:
			pass
	return populate_res(mu, np.array(ests1), np.array(ests2), np.array(ests5))


def populate_res(mu, ests1, ests2, ests5):
	print(np.mean(ests1))
	print(np.var(ests1))
	print("######")
	print(np.mean(ests2))
	print(np.var(ests2))
	print("######")
	print(np.mean(ests5))
	print(np.var(ests5))
	plt.hist(ests1)
	res = np.zeros((3,3))
	populate_arr(0, ests1, mu, res)
	populate_arr(1, ests2, mu, res)
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
    parser.add_argument('lmb', type=int, help='scale of exponential distribution for sampling time between events')
    parser.add_argument('mu', type=int, help='scale of exponential distribution for sampling duration time of events')
    parser.add_argument('ttx', type=bool, help='True if estimating TTX, False if estimating rate')
    args = parser.parse_args()
    s, e, lmb, mu, ttx = args.s, args.e, args.lmb, args.mu, args.ttx

    # run simulations and compare TTR estimates 
    cmp_ests(s=s, e=e, lmb=lmb, mu=mu, ttx=ttx)