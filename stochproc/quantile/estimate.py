import numpy as np


def est_1(a, p):
    #TODO: replace this with quick find.
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n))
    g = n*p-j
    gamma = 1-(g==0)*1.0
    # The indexing here is one off from the R documentation because
    # python starts at 0
    return a[j-1]*(1-gamma)+a[j]*gamma


def tst_est_1():
    assert est_1([1,2,3,4,5],.60)==3.0
    assert est_1([1,2,3,4,5],.59)==3.0
    assert est_1([1,2,3,4,5],.61)==4.0


def est_2(a, p):
    #TODO: replace this with quick find.
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n))
    g = n*p-j
    if g==0:
        gamma = 0.5
    else:
        gamma = 1
    # The indexing here is one off from the R documentation because
    # python starts at 0
    return a[j-1]*(1-gamma)+a[j]*gamma


def tst_est_2():
    assert est_1([1,2,3,4,5],.60)==3.0
    assert est_1([1,2,3,4,5],.59)==3.5
    assert est_1([1,2,3,4,5],.61)==4.0


def est_3(a, p):
    #TODO: replace this with quick find.
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n))
    g = n*p-j
    if g == 0 and j%2 == 0:
        gamma = 0
    else:
        gamma = 1
    # The indexing here is one off from the R documentation because
    # python starts at 0
    return a[j-1]*(1-gamma)+a[j]*gamma


def est_4(a, p):
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n))
    gamma = n*p-j
    return a[j-1]*(1-gamma)+a[j]*gamma


def est_5(a, p):
    m = 0.5
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n+m))
    gamma = n*p-j
    return a[j-1]*(1-gamma)+a[j]*gamma+m


def est_6(a, p):
    m = p
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n+m))
    gamma = n*p-j
    return a[j-1]*(1-gamma)+a[j]*gamma+m


def est_7(a, p):
    return np.percentile(a, p*100)


def est_8(a, p):
    m = (p+1)/3
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n+m))
    gamma = n*p-j
    return a[j-1]*(1-gamma)+a[j]*gamma+m


def est_9(a, p):
    m = p/4+3/8
    a = sorted(a)
    n = len(a)
    j = int(np.floor(p*n+m))
    gamma = n*p-j
    return a[j-1]*(1-gamma)+a[j]*gamma+m


# References
# [1] https://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html
