import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt


def estimate_median(x, is_sorted=False):
    n = len(x)
    if not is_sorted:
        x = sorted(x)
    # For odd n, both ceiling and floor will be the median.
    # For even n, we'll end up averaging two numbers.
    lo = int(np.floor((n+1)/2))
    hi = int(np.ceil((n+1)/2))
    x1 = x[lo]
    x2 = x[hi]
    return (x1+x2)/2


def plot_bias():
    for sampl in np.arange(35, 220, 5):
        errs = []
        ests = []
        real_val = lognorm.ppf(0.5, 1, 0)
        for _ in range(100000):
            x = lognorm.rvs(1, 0, size=sampl)
            est_val = estimate_median(x)
            err = (real_val-est_val)/real_val
            errs.append(err)
            ests.append(est_val)

        print(np.mean(errs))

        plt.hist(ests, bins=np.arange(0, 4, .1))
        plt.axvline(real_val, label="actual median", color="black")
        plt.axvline(np.mean(ests),
                    label="avg estimated value of median on sample size: "
                    + str(sampl), color="purple")
        plt.legend()
        plt.title("Sample size = " + str(sampl))
        plt.savefig('plots/sample_' + str(sampl) + '.png')
        plt.close()
        print('processed sample size ' + str(sampl))


if __name__ == "__main__":
    plot_bias()
