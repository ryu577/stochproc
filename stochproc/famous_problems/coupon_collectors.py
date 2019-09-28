import numpy as np

def double_harmonic(n):
    res = 0
    for k in range(1,n+1):
        for j in range(1,k+1):
            res += 1/j/k
    return res

def s_harmonic(n):
    res = 0
    for k in range(1,n+1):
        res += 1/k
    return res

def s_harmonic_sq(n):
    res = 0
    for k in range(1,n+1):
        res += 1/k/k
    return res

def var1(n):
    return n**2*s_harmonic_sq(n)-n*s_harmonic(n)

def var2(n):
    e_t = n*s_harmonic(n)
    return 2*n**2*double_harmonic(n)\
            -e_t-e_t**2

