import numpy as np
import pandas as pd
from scipy.special import comb


def single_machine_transitions(t, lmb=1/10, mu=1/10):
	"""
	Generates the raw state transition data for 
	a machine with failure rate lmb and repair rate mu.
	"""
	cols=['start','end','duration','status']
	dat_lst = []
	running_time = 0
	while True:
		up_time = np.random.exponential(1/lmb)
		dat_lst.append([running_time, running_time+up_time, up_time, 0])
		running_time += up_time
		if running_time > t:
			break
		down_time = np.random.exponential(1/mu)
		dat_lst.append([running_time, running_time+down_time, down_time, 1])
		running_time += down_time
		if running_time > t:
			break
	return pd.DataFrame(dat_lst, columns=cols)


def state_change_times(n=3):
    """
    Returns the times at which the state of the k-of-n system change
    as a data frame. At the start of a new downtime, the number of
    components down decreases by 1 and at the end of a new downtime,
    it increases by 1.
    """
    res = pd.DataFrame(columns=['time','status'])
    times = np.array([])
    statuses = np.array([])
    total_durtn = 0
    for _ in range(n):
        df = single_machine_transitions(1000,1/10,1/10)
        total_durtn += sum(df['duration'])
        df = df[df['status']==1]
        times = np.concatenate((times,df['start']),axis=0)
        statuses = np.concatenate((statuses,np.ones(len(df['start']))),axis=0)
        times = np.concatenate((times,df['end']),axis=0)
        statuses = np.concatenate((statuses,-1*np.ones(len(df['start']))),axis=0)
    res['time'] = times
    res['status'] = statuses
    res = res.sort_values(by='time')
    return res, total_durtn


def n_machine_transitions(n=3):
    """
    Converts the transition data for the individual machines into 
    the transition data for the entire system. Here, states are defined
    as the number of machines down at any given time.
    """
    trnstns, durtn = state_change_times(n)
    rows = len(trnstns)
    out = pd.DataFrame(columns=['start', 'end', 'down_vms'])
    out['down_vms'] = np.cumsum(trnstns['status'])[:rows-1]#Remove the last row.
    out['start'] = np.array(trnstns['time'][:rows-1])
    out['end'] = np.array(trnstns['time'][1:])
    return out, durtn


def count_intervals(dat):
    """
    We will have instances of continuous downtimes being split into different
    state transitions. For example, if a 2 out of 3 system transitions
    from 2 down to 3 down and then back to 2 down, we want to treat those
    three transitions as a single DOWN event for the system. This method helps with
    that.
    """
    prev_start = 0
    prev_end = 0
    intervals = 0
    for _, row in dat.iterrows():
        curr_start = row['start']
        curr_end = row['end']
        if curr_start > prev_end:
            intervals += 1
        prev_start = curr_start
        prev_end = curr_end
    return intervals


def k_of_n_system(k=2, n=3):
    """
    Calculates the simulated interruption rate for a k-out-of-n
    system.
    """
    dat, durtn = n_machine_transitions(n)
    sys_down = dat[dat['down_vms']>=2]
    downs = count_intervals(sys_down)
    # We took total duration here, so time system was measured is
    # the average of those durations.
    rate = downs/(durtn/n)
    return rate


def closed_form(k=2,n=2,lmb=0.1,mu=0.1):
    return n*lmb**k*mu**(n-k+1)/(lmb+mu)**n*comb(n-1,k-1)


def main():
    """
    Verifies the closed form interruption rate for a k-of-n
    system matches the simulated hazard rate for the same system.
    """
    closed_form_rate = closed_form()
    simulated_rate = k_of_n_system()
    print("The closed form gives us:"+str(closed_form_rate))
    print("And the simulated result is:"+str(simulated_rate))

