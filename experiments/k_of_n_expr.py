import numpy as np
from stochproc.birth_death_processes.reliability.k_of_n import k_of_n_av, k_of_n_rate
import matplotlib.pyplot as plt

def plot_air_systems():
    lmb=3
    mus = np.arange(1,20,1)
    for n in np.arange(3,15,2):
        rel_rates=[]
        for mu in mus:
            rel_rates.append(k_of_n_rate(n//2+1,n,lmb,mu)/lmb)
        plt.plot(mus,rel_rates,label="n="+str(n))
    plt.legend()
    plt.show()

