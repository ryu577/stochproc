import numpy as np
from stochproc.birth_death_processes.cad_simulator import birth_death_gen
from algorith.arrays.cad_as_violations import complete_intervals
import pandas as pd
from scipy.stats import binom

def tst_availability():
    lmb=.1
    mu=.2
    node1 = birth_death_gen(lmb,mu,1e4)
    node2 = birth_death_gen(lmb,mu,1e4)
    node3 = birth_death_gen(lmb,mu,1e4)

    node1 = node1[node1.start>7e3]
    node2 = node2[node2.start>7e3]
    node3 = node3[node3.start>7e3]

    node_dat = pd.concat([node1,node2,node3])
    node_dat = node_dat[node_dat.state=="down"]
    node_dat = node_dat.sort_values(by=['start'])

    dd = complete_intervals(node_dat)
    dd=dd[dd.vms_down>=2]
    down_durtn = sum(dd.end-dd.start)
    print("Simulated availability: " + str(1-down_durtn/3000))

    ### Closed form..
    p = mu/(lmb+mu)
    av = binom.pmf(2,3,p)+binom.pmf(3,3,p)
    print("Closed form availability: "+ str(av))


