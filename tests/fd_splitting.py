import numpy as np
import matplotlib.pyplot as plt


## Availability for architectures.
def fn_right(p):
    return (3*p**2*(1-p)+p**3)**2

def fn_left(p):
    return 20*p**3*(1-p)**3+15*p**4*(1-p)**2+6*p**5*(1-p)+p**6


ps = np.arange(0,1,0.01)
rights = fn_right(ps)
lefts = fn_left(ps)

plt.plot(ps,1-rights,label="right model")
plt.plot(ps,1-lefts,label="left model")
plt.legend()
plt.show()

