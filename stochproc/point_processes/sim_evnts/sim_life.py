import numpy as np
import matplotlib.pyplot as plt
import argparse


def est_mean(s, e, lmb=1, mu=1):
	t = 0
	est1 = 0
	est2 = 0
	est3 = 0
	n = 0
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
			est3 = est3 + durtn
		elif e1 > e and s1 < e:
			est2 = est2 + e - max(s, s1)
			est3 = est3 + e - s1
	res1 = est1/n
	res2 = est2/n
	res3 = est3/n
	return res1, res2, res3


# Tests..
# est3 >= est1.
def cmp_ests(s=100, e=120, lmb=1, mu=1):
	ests1 = []; ests2 = []; ests3 = []
	for i in range(2000):
		try:
			est1, est2, est3 = est_mean(s, e, lmb, mu)
			ests1.append(est1)
			ests2.append(est2)
			ests3.append(est3)
		except:
			pass
	print(np.mean(ests1))
	print(np.var(ests1))
	print("######")
	print(np.mean(ests2))
	print(np.var(ests2))
	print("######")
	print(np.mean(ests3))
	print(np.var(ests3))
	plt.hist(ests1)
	#plt.avhline(1)
	# plt.show()

if __name__=='__main__':
    # parse arguments 
    parser = argparse.ArgumentParser(description='Run simulations on incident \
        TTRs (time to resolution)')
    parser.add_argument('s', type=int, help='start time of window')
    parser.add_argument('e', type=int, help='end time of window')
    parser.add_argument('lmb', type=int, help='scale of exponential \
        distribution for sampling time between incident arrivals')
    parser.add_argument('mu', type=int, help='scale of exponential distribution\
        for sampling duration time of incidents')
    args = parser.parse_args()
    s, e, lmb, mu = args.s, args.e, args.lmb, args.mu

    # run simulations and compare TTR estimates 
    cmp_ests(s=s, e=e, lmb=lmb, mu=mu)