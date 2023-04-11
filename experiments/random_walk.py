import numpy as np


def sim_walk(p=.9, ceil=1e3):
	x = 0
	if np.random.uniform() < p:
		x = x + 1
	else:
		x = x - 1
	while x != 0:
		if np.random.uniform() < p:
			x = x + 1
		else:
			x = x - 1
		if abs(x) > ceil:
			return 0
	return 1
