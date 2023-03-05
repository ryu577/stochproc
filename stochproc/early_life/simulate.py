import numpy as np
import matplotlib.pyplot as plt


t2s = []
t3s = []

for i in range(100000):
	# How long does the planet stay habitable?
	t1 = 5
	# How long does it take for life to evolve
	t2 = np.random.exponential(60)
	# Next instance of abiogenesis
	t2prime = np.random.exponential(60)
	# How long for intelligence to evolve?
	t3 = np.random.exponential(60)
	if t2+t3 < t1 and t2prime+t2>4.5:
		t2s.append(t2)
		t3s.append(t3)

print(np.mean(t2s))
print(np.mean(t3s))
plt.hist(t2s)
plt.show()

