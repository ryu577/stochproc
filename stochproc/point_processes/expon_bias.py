import numpy as np
import matplotlib.pyplot as plt

n_sim=10000; lm_hat=0
for _ in range(n_sim):
    t=0; n=0
    while t<10:
        t+=np.random.exponential(1)
        if t<10:
            n+=1
    lm_hat += n/10

print(lm_hat/n_sim)
