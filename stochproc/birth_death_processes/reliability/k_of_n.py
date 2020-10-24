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

## Doesn't seem to work for 1 of 2 system.
def k_of_n_repair_rate(k=2,n=3,lmb=2,mu=6):
    ## Per equation 9.33 of Ross
    p=mu/(lmb+mu)
    av=k_of_n_av(k,n,p)
    return (1-av)*k_of_n_rate(k,n,lmb,mu)/av

def k_of_n_sim(k=2,n=3,lmb=2,mu=6):
    nodes = []
    for _ in range(n):
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

########################################
#### Some simple cases.

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

def tst_2_of_2():
    node1 = birth_death_gen(1,2,1e4)
    node1 = node1[(node1.start>7e3) & (node1.state=="down")]
    node2 = birth_death_gen(1,2,1e4)
    node2 = node2[(node2.start>7e3) & (node2.state=="down")]
    dat = pd.concat([node1,node2])
    dat = dat.sort_values(by=['start'])

    res=complete_intervals(dat)
    downs = interrupts(res.down,1)
    down_durtns = (res.end-res.start)[res.down>=1]
    down_durtn = sum(down_durtns)
    lmb = downs/(3000-down_durtn)
    mu = downs/down_durtn
    print(lmb)
    print(mu)

