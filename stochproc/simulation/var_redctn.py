import numpy as np
from scipy.stats.distributions import norm

"""
Reducing variance using importance sampling.
"""


print("Probability that std normal will be greater than 2 is:" + str((1-norm.cdf(2,0,1))))

print("What we get from direct simulation:" + str(sum(np.random.normal(0,1,size=10000) > 2)/10000))

summ = 0
for x in np.random.normal(2,1,size=10000):
	summ += (x>2)*norm.pdf(x,0,1)/norm.pdf(x,2,1)

print("With importance sampling:" + str(summ/10000))


