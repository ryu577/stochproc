import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


def prcntl_bias(q, n, interpolate=1):
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)
    if interpolate==1:
        summ += np.modf(q*(n-1))[0]/(n-lt-1)
    elif interpolate==2:
        summ += expon_frac(q, n)/(n-lt-1)
    elif interpolate==3:
        summ += 1/(n-lt-1)
    elif interpolate==4:
        summ += expon_low_mse_frac(q,n)/(n-lt-1)
    return -np.log(1-q)-summ


def prcntl_var(q, n, interpolate=1):
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)**2
    if interpolate==1:
        summ += (np.modf(q*(n-1))[0]/(n-lt-1))**2
    elif interpolate==2:
        summ += (expon_frac(q, n)/(n-lt-1))**2
    elif interpolate==3:
        summ += (1/(n-lt-1))**2
    elif interpolate == 4:
        summ += (expon_low_mse_frac(q,n)/(n-lt-1))**2
    return summ


def expon_frac(q, n):
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)
    return (-np.log(1-q)-summ)*(n-lt-1)


def expon_low_mse_frac(q, n):
    lt = int(np.floor(q*(n-1)))
    return (lt-1)/2*prcntl_bias(q,n,0)


def analyze_bias(n=55, save_close=True):
    qs = np.arange(0.01,1.0,0.01)
    # biases for linear interpolation.
    biases1 = [prcntl_bias(q,n,1) for q in qs]
    # biases for no interpolation
    biases2 = [prcntl_bias(q,n,0) for q in qs]
    # biases for low bias strategy: its always zero. So we over-write it.
    biases3 = [prcntl_bias(q,n,2) for q in qs]
    # biases for other no interpolation strategy.
    biases3 = [prcntl_bias(q,n,3) for q in qs]
    # st deviations for linear interpolation
    st_devs = [np.sqrt(prcntl_var(q,n,1)) for q in qs]
    st_devs1 = [np.sqrt(prcntl_var(q,n,2)) for q in qs]
    biases4 = [prcntl_bias(q,n,4) for q in qs]
    st_devs2 = [np.sqrt(prcntl_var(q,n,4)) for q in qs]

    plt.style.use('dark_background')
    plt.plot(qs, biases1, label="Linear interpolation strategy")
    plt.plot(qs, biases2, label="No interpolation strategy")
    plt.plot(qs, biases3, label="No interpolation strategy other")
    plt.plot(qs, st_devs, label="st_dev of linear interpolation")
    plt.plot(qs, st_devs1, label="st_dev of low bias")
    plt.plot(qs, biases4, label="Bias of low MSE strategy")
    plt.plot(qs, st_devs2, label="st dev of low MSE strategy")

    #plt.plot(np.arange(0.1,1.0,0.01), biases3)
    plt.axhline(0, color="white")
    fn1 = lambda q: prcntl_bias(q, n)
    rt = root_scalar(fn1,bracket=[0,1],method='bisect').root
    print("Unbiased percentile: " + str(rt))
    plt.axvline(rt,color="white")
    plt.title("Sample size n=" + str(n) + " \nPercentile with no bias is: " + str("{:.2f}".format(rt*100)))
    plt.xlabel("Percentile (q)")
    plt.ylabel("Bias for the exponential distribution")
    plt.legend()
    if save_close:
        plt.savefig('plots/sample_' + str(n) + '.png')
        plt.close()


for n in range(15,405,25):
    analyze_bias(n)

