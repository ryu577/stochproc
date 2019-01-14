import numpy as np
import pandas as pd


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
	res = pd.DataFrame(columns=['time','status'])
	times = np.array([])
	statuses = np.array([])
	for i in range(n):
		df = single_machine_transitions(1000,1/10,1/10)
		df = df[df['status']==1]
		times = np.concatenate((times,df['start']),axis=0)
		statuses = np.concatenate((statuses,np.ones(len(df['start']))),axis=0)
		times = np.concatenate((times,df['end']),axis=0)
		statuses = np.concatenate((statuses,-1*np.ones(len(df['start']))),axis=0)
	res['time'] = times
	res['status'] = statuses
	res = res.sort_values(by='time')
	return res


def n_machine_transitions(n=3):
	trnstns = state_change_times(n)
	rows = len(trnstns)
	out = pd.DataFrame(columns=['start', 'end', 'down_vms'])
	out['down_vms'] = np.cumsum(trnstns['status'])[:rows-1]#Remove the last row.
	out['start'] = np.array(trnstns['time'][:rows-1])
	out['end'] = np.array(trnstns['time'][1:])
	return out


def k_of_n_system(k=2,n=3):
    dat = n_machine_transitions(n)
    
    

