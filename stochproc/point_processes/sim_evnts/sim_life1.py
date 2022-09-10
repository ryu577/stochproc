# import libraries 
import argparse
import numpy as np 

# compute E(TTR) baseline and estimators 
def est_mean(s, e, lmb=1, mu=1):
    '''
    Performs a simulation of incidents arriving from time t=0 to time 
    e + 100. Computes baseline E(TTR) (TTR = time to resolution), 
    estimator as well as other estimators for E(TTR) 

    Parameters: 
    s (int) - start time of time window 
    e (int) - end time of time window 
    lmb (int) - scale of exponential distribution for sampling time 
    between incident arrivals
    mu (int) - scale of exponential distribution for sampling duration
    time of incidents 
    
    Returns: 
    tuple of estimators - tuple[int]
    (est0, est1, est2, est3, est4, est5, est6)

    est0 (float) - first/baseline E(TTR) estimator 
    est0 = sum(e_i - s_i) / n0, where s < s_i < e_i < e and n0 = number of 
    incidents where the constraint is met 

    est1 (float) - second E(TTR) estimator 
    est1 = sum(e_i - s_i) / n1 where s < e_i < e and n1 = number of incidents
    where the constraint is met 

    est2 (float) - third E(TTR) estimator 
    est2 = sum(d_i) / n2 where n2 = number of incidents where s_i < e < e
    d_i = {
        e_i - max(s, s_i), when s < e_i < e
        e - max(s, s_i), when s_i < e < e_i 
    }

    est3 (float) - fourth E(TTR) estimator 
    est3 = sum(d_i) / n3 where n3 = number of incidents where s < e_i < e
    d_i = {
        e_i - s_i, where s < e_i < e
        e - s_i, where s_i < e < e_i
    }

    est4 (float) - fifth E(TTR) estimator 
    est4 = est4_1 + est4_2 
    est4_1 = sum(e_i - max(s, s_i)) / n4_1 where s < e_i < e and n4_1 = number
    of incidents where the constraint is met 
    est4_2 = sum(e - max(s, s_i)) / n4_2 where s_i < e < e_i and n4_2 = number
    of incidents where the constraint is met 

    est5 (float) - sixth E(TTR) estimator 
    est5 = est5_1 + est5_2 
    est5_1 = sum(e_i - s_i) / n5_1 where s < e_i < e and n5_1 = number of 
    incidents where the constraint is met 
    est5_2 = sum(e - s_i) / n5_2 s_i < e < e_i and n5_2 = number of incidents
    where the constriant is met 

    '''
    # initialize simulation 
    t = 0
    est0, n0 = 0, 0 
    est1, n1 = 0, 0 
    est2, n2 = 0, 0 
    est3, n3 = 0, 0 
    est4_1, est4_2, n4_1, n4_2 = 0, 0, 0, 0
    est5_1, est5_2, n5_1, n5_2 = 0, 0, 0, 0
    
    # run simulation, introduce incidents and compute estimators 
    while t < e + 100: 
        # sample time difference between incident arrivals from 
        # exponential distribution and update time 
        t_del = np.random.exponential(lmb)
        t = t + t_del

        # sample time duration of incident from exponential distribution 
        # set start and end times of incident 
        s_i = t
        durtn = np.random.exponential(mu)
        e_i = t + durtn

        # compute est0 
        if s < s_i and e_i < e: 
            est0 += e_i - s_i
            n0 += 1
        
        # compute est1 
        if s < e_i and e_i < e: 
            est1 += e_i - s_i 
            n1 += 1 

        # compute est2
        if s < e_i and e_i < e:
            est2 += e_i - max(s, s_i)
            n2 += 1 
        elif s_i < e and e < e_i: 
            est2 += e - max(s, s_i)
        
        # compute est3
        if s < e_i and e_i < e: 
            est3 += e_i - s_i 
            n3 += 1 
        elif s_i < e and e < e_i: 
            est3 += e - s_i

        # compute est4
        if s < e_i and e_i < e: 
            est4_1 += e_i - max(s, s_i)
            n4_1 += 1 
        elif s_i < e and e < e_i: 
            est4_2 += e - max(s, s_i)
            n4_2 += 1
        
        # compute est5
        if s < e_i and e_i < e: 
            est5_1 += e_i - s_i 
            n5_1 += 1
        elif s_i < e and e < e_i: 
            est5_2 += e - s_i 
            n5_2 += 1
        
    # finalize computations 
    est0 /= n0 
    est1 /= n1 
    est2 /= n2 
    est3 /= n3 
    est4 = (est4_1 + est4_2) / (n4_1 + n4_2)
    est5 = (est5_1 + est5_2) / (n5_1 + n5_2)

    # return E(TTR) estimators 
    return est0, est1, est2, est3, est4, est5


def cmp_ests(s=100, e=120, lmb=1, mu=1):
    # initialize stores for computed estimators 
    ests0 = []; ests1 = []; ests2 = []; ests3 = []; ests4 = []; ests5 = []
    for _ in range(2000):
        try: 
            est0, est1, est2, est3, est4, est5 = est_mean(s, e,\
                lmb, mu)
            ests0.append(est0)
            ests1.append(est1)
            ests2.append(est2)
            ests3.append(est3)
            ests4.append(est4)
            ests5.append(est5)
        except: 
            pass 

    # compute and print estimator distribution info 
    ests = [ests0, ests1, ests2, ests3, ests4, ests5]
    dist_info = []
    for i in range(len(ests)):
        if i == 0: 
            print(f'Estimator {i} (Baseline)')
        else: 
            print(f'Estimator {i}')
        mean = np.mean(ests[i])
        variance = np.var(ests[i])
        dist_info.append(mean)
        dist_info.append(variance)
        print(f'E(TTR): {mean}')
        print(f'Var(TTR): {variance}')
        print('######')
    return dist_info


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