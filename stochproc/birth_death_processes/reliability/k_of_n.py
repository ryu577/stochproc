import numpy as np
import pandas as pd
from scipy.stats import binom

from stochproc.birth_death_processes.birth_death_gen import birth_death_gen
from algorith.arrays.birth_death.cad_as_violations import complete_intervals, system_down_intervals


def k_of_n_av(k,n,p):
    return binom.sf(k-1,n,p)

def k_of_n_rate(k=2,n=3,lmb=2,mu=6):
    p=mu/(lmb+mu)
    return n*lmb*p*binom.pmf(k-1,n-1,p)/binom.sf(k-1,n,p)

def k_of_n_sim(k=2,n=3,lmb=2,mu=6):
    nodes = []
    for i in range(n):
        node1 = birth_death_gen(lmb,mu,1e4)
        node1 = node1[(node1.start>7e3) & (node1.state=="down")]
        nodes.append(node1)
    dat = pd.concat(nodes)
    dat = dat.sort_values(by=['start'])
    dat1 = system_down_intervals(np.array(dat.start),\
                    np.array(dat.end),n-k+1)
    interruptions = sum(dat1.down)
    dat1 = dat1[dat1.down==1]
    down_time = sum(dat1.end-dat1.start)
    up_time=3000-down_time
    rate = interruptions/(up_time)
    av = up_time/3000
    return av, rate


def tst_av_2_of_3():
    """
    Check that the simulated availability of a 2-of-3
    system matches the closed form expression.
    """
    lmb=2
    mu=6
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
    down_evnts = 0
    for i in range(len(dd)):
        if dd.vms_down[i] == 2:
            if dd.vms_down[i-1]<2:
                down_evnts+=1

    #up_durtn = sum(ups.end-ups.start)
    dd=dd[dd.vms_down>=2]
    down_durtn = sum(dd.end-dd.start)
    up_durtn = 3000-down_durtn
    print("Simulated availability: " + str(1-down_durtn/3000))
    print("Failure rate: " + str(down_evnts/up_durtn))

    ### Closed form..
    p = mu/(lmb+mu)
    av = k_of_n_av(2,3,p)
    rate = k_of_n_rate(2,3,lmb,mu)
    print("Closed form availability: "+ str(av))
    print("Closed form rate: " + str(rate))

