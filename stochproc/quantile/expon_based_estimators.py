
import numpy as np


def expon_frac(q, n):
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)
    return (-np.log(1-q)-summ)*(n-lt-1)


def prcntl(a, q, interpolate=2):
    a = sorted(a)
    n = len(a)
    lt = int(np.floor(q*(n-1)))
    frac = 0
    if interpolate == 1:
        frac = np.modf(q*(n-1))[0]
    elif interpolate == 2:
        frac = expon_frac(q, n)
    return a[lt]*(1-frac)+a[lt+1]*frac


