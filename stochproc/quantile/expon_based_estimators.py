import numpy as np


def expon_frac(q, n):
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)
    return (-np.log(1-q)-summ)*(n-lt-1)


# TODO: Add arxiv link for this.
def expon_fracs(q, n):
    lt = int(np.floor(q*(n-1)))
    summ = 0
    for ix in range(lt+1):
        summ += 1/(n-ix)
    beta = np.log(1-q) + summ
    f = -(beta/2)
    g = -(beta/2)*(n-lt-2)
    return f, g


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


def prcntl2(a, q):
    a = sorted(a)
    n = len(a)
    lt = int(np.floor(q*(n-1)))
    if lt+2 > len(a)-1:
        return prcntl(a, q, 2)
    else:
        f, g = expon_fracs(q, n)
        return a[lt]*(1-f-g)+a[lt+1]*f+a[lt+2]*g


def prcntl3(a, q):
    n = len(a)
    summ = 0
    for ix in range(n):
        summ += 1/(n-ix)
    return np.mean(a)*(-np.log(1-q))


def prcntl4(a, q):
    a = sorted(a)
    n = len(a)
    lt = int(np.floor(q*(n-1)))
    i = lt+1
    summ = 0
    for j in range(i):
        summ += 1/(n-j)
    b = summ + np.log(1-q)
    u = (1+b)*a[i-1]
    summ2 = 0
    for j in range(i, n):
        summ2 += a[j]
    u = u - b/(n-i)*summ2
    return u


def prcntl5(a, q):
    a = sorted(a)
    n = len(a)
    i = 2
    summ = 0
    for j in range(i):
        summ += 1/(n-j)
    b = summ + np.log(1-q)
    u = (1+b)*a[i-1]
    summ2 = 0
    for j in range(i, n):
        summ2 += a[j]
    u = u - b/(n-i)*summ2
    return u
