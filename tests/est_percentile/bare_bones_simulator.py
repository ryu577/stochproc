import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# A function that is going to generate random variables for the Lognormal distribution.
def rvs_fn(n):
    return lognorm.rvs(1, 0, size=n)

# A function that will tell you the true q-percentile for the Lognormal distribution with q being an input.
def ppf_fn(q):
    return lognorm.ppf(q, 1, 0)

# The number of samples we'll be collecting in each parallel universe.
n = 15
# The percentiles we're going to consider from the 5th percentile
# to the 95th percentile.
qs = np.arange(0.05, 1, 0.05)
per_percentile_biases = []
# Loop through the q's (percentiles)
for q in qs:
    # The true percentile of the distribution
    # generating the data.
    real_percentile = ppf_fn(q)
    errs = []
    # Loop through a large number of parallel universes, m.
    # the rows of the matrix.
    for _ in range(10000):
        # Generate n samples for each parallel universe.
        # the columns of the matrix.
        x = rvs_fn(n)
        # Estimate the percentile from the finite sample.
        # The inbuilt numpy method uses the linear interpolation strategy.
        estimated_percentile = np.percentile(x, q*100)
        # The difference between the real and estimated values is the error.
        error = (real_percentile - estimated_percentile)
        # Collect the error into an array, one error for each parallel universe.
        errs.append(error)
    # The average value of the errors is called the bias. We do this for each percentile.
    per_percentile_biases.append(np.mean(errs))


# Now plot the bias for each percentile
plt.plot(qs, per_percentile_biases)
plt.show()
