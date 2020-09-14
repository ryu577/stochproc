import numpy as np
import pandas as pd
from stochproc.birth_death_processes.birth_death_gen import birth_death_gen


def state_change_times(n=3):
	res = pd.DataFrame(columns=['time','status'])
	times = np.array([])
	statuses = np.array([])
	for i in range(n):
		df = birth_death_gen(1/10,1/10,1000)
		df = df[df['state']=="up"]
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
    system_downs = dat[dat['down_vms']>=2]
    
    

