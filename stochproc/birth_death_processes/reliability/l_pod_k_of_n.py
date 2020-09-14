import numpy as np
import pandas as pd
from algorith.arrays.num_systems.dynamic_base import GenBase, to_binary
from scipy.special import comb
from scipy.stats import binom

from stochproc.birth_death_processes.birth_death_gen import birth_death_gen
from algorith.arrays.birth_death.cad_as_violations import \
        complete_intervals, system_down_intervals, num_down_w_times_v1


def l_pod_k_of_n_av(l,n,k,p,q,pod_less_mcs=0):
    """
    See here: https://math.stackexchange.com/questions/3825082/reliability-of-an-l-pod-k-of-n-system?noredirect=1#comment7888736_3825082
    Answer matches with legacy method: pffc_resiliency(3,7,.9,.8,1)==l_pod_k_of_n(3,7,4,0.9,0.8)
    args:
        pod_less_mcs: This is a parameter for the AIR formula. 
                Machines for whom the pod is garunteed to work.
    """
    ceil=np.ceil(n/l)
    flr=np.floor(n/l)
    ## Num of hero and joe pods.
    # A hero pod has one more machine 
    # than a joe pod.
    h=int(n-l*flr)
    j=int(l*ceil-n)
    prob=0.0
    ## All combinations of hero and joe pod availability.
    for h1 in range(h+1):
        for j1 in range(j+1):
            #Num of available machines with these many pods.
            nn = h1*ceil+j1*flr+pod_less_mcs
            if nn>=k:
                ## We get a k of nn system among the machines.
                prob+=binom.pmf(h1,h,q)*binom.pmf(j1,j,q)*binom.sf(k-1,nn,p)
    return prob


def l_pod_k_of_n_rate(l,n,k,lmb_mc,mu_mc,lmb_pod,mu_pod):
    p=mu_mc/(mu_mc+lmb_mc)
    q=mu_pod/(mu_pod+lmb_pod)
    ceil=np.ceil(n/l)
    flr=np.floor(n/l)
    ## Num of hero and joe pods.
    # A hero pod has one more machine 
    # than a joe pod.
    h=int(n-l*flr)
    j=int(l*ceil-n)
    #Total number of machines in hero and
    # joe pods.
    n_h=h*ceil
    n_j=j*flr
    # Numerator of the AIR term.
    #First, the hero pods
    numr1 = h*(l_pod_k_of_n_av(l-1,n-ceil,k,p,q,ceil)-\
                l_pod_k_of_n_av(l-1,n-ceil,k,p,q))
    #Next, the joe pods
    numr2 = j*(l_pod_k_of_n_av(l-1,n-flr,k,p,q,flr)-\
                l_pod_k_of_n_av(l-1,n-flr,k,p,q))
    #Next, the hero machines
    numr3 = n_h*(l_pod_k_of_n_av(l,n-1,k-1,p,q)-\
                l_pod_k_of_n_av(l,n-1,k,p,q))
    #Next, the joe machines
    ## First, availability when this Joe M/C is up.
    # Note that the second part is not needed since 
    # it cancels out with the next term.
    joe_mc_up = q*l_pod_k_of_n_av(l-1,n-flr,k,p,q,flr)
    #           +(1-q)*l_pod_k_of_n_av(l-1,n-flr,k,p,q)
    #
    ## Next, the availability when this Joe M/C is down.
    joe_mc_down = q*l_pod_k_of_n_av(l-1,n-flr,k,p,q,flr-1)
    #            +(1-q)*l_pod_k_of_n_av(l-1,n-flr,k,p,q)
    #
    numr4 = n_j*(joe_mc_up-joe_mc_down)
    ## Now the final numerator:
    numr=q*lmb_pod*(numr1+numr2)+p*lmb_mc*(numr3+numr4) 
    ## Now the denominator, which is just reliability
    denom = l_pod_k_of_n_av(l,n,k,p,q)
    return numr/denom

def l_pod_k_of_n_legacy(n=7, l=3, k=4):
    p_n=0.9; p_r=0.9
    h,j,alpha,beta=striping(l,n)
    machines_per_pod = np.array([(alpha if i<h else beta) for i in range(int(h+j))])
    arr = GenBase(machines_per_pod)
    tot = int(np.prod(arr.bases+1))
    prob = 0.0
    for _ in range(tot-1):
        arr.add_one()
        up_nodes = sum(arr.arr_vals)
        binoms = comb(arr.bases,arr.arr_vals)
        wt = np.prod(binoms)*p_n**(up_nodes)*(1-p_n)**(n-up_nodes)
        for j in range(2**l):
            aa = to_binary(j, n)
            if sum(arr.arr_vals*aa) >= k:
                rows_up = sum(aa)
                wts = arr.arr_vals[aa>0]
                wts = wts[wts>0] ##Controvertial.
            prob += wt*p_r**rows_up*(1-p_r)**(n-rows_up)
    return prob


def striping(l=3, n=7):
    """
    Given the number of PF-rows and replicas, uniformly 
    distributes the PF-rows among the replicas. With a uniform
    striping, we will have two kinds of rows, one kind having
    one more replica than the other kind. This method returns
    the number of rows of each kind (h and j) and the number
    of replicas in each of the row-kinds.
    """
    alpha = np.ceil(float(n)/l)
    beta = alpha-1
    h = n-l*(alpha-1)
    j = l*alpha-n
    return (h,j,alpha,beta)


####################################
## Unverified
### TODO: Verify this works.
def l_pod_k_of_n_sim(l,n,k,lmb_mc,mu_mc,lmb_pod,mu_pod):
    ceil=np.ceil(n/l)
    flr=np.floor(n/l)
    h=int(n-l*flr)
    j=int(l*ceil-n)
    nodes = []
    for _ in range(h):
        pod_nodes = sim_one_pod(ceil,lmb_mc,mu_mc,lmb_pod,mu_pod)
        nodes.append(pod_nodes)
    dat_h = pd.concat(nodes)
    dat_h = dat_h.sort_values(by=['start'])
    for _ in range(j):
        pod_nodes = sim_one_pod(flr,lmb_mc,mu_mc,lmb_pod,mu_pod)
        nodes.append(pod_nodes)
    dat_j = pd.concat(nodes)
    dat_j = dat_j.sort_values(by=['start'])
    dat=pd.concat([dat_h,dat_j])
    ds,ts=num_down_w_times_v1(dat.start,dat.end,dat.down)


### TODO: Verify this works.
def sim_one_pod(machines,lmb_mc,mu_mc,lmb_pod,mu_pod):
    nodes = []
    for _ in machines:
        node1=birth_death_gen(lmb_mc,mu_mc,1e4)
        node1 = node1[(node1.start>7e3) & (node1.state=="down")]
        nodes.append(node1)
    dat = pd.concat(nodes)
    dat = dat.sort_values(by=['start'])
    dat1 = complete_intervals(dat)
    ##Now simulate the pod..
    pod1=birth_death_gen(lmb_pod,mu_pod,1e4)
    pod1 = pod1[(pod1.start>7e3) & (pod1.state=="down")]
    pod1["down"] = machines
    pod_dat = pd.concat([dat1,pod1])
    pod_dat.sort_values(by=['start'])
    ds,ts=num_down_w_times_v1(pod_dat.start,pod_dat.end,pod_dat.down)
    return pd.DataFrame({'start':ts[:len(ts)-1],'end':ts[1:],'down':ds})

