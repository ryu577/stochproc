import numpy as np
import pandas as pd
from algorith.arrays.num_systems.dynamic_base import GenBase, to_binary
from scipy.special import comb
from scipy.stats import binom

from stochproc.birth_death_processes.birth_death_gen import birth_death_gen
from stochproc.birth_death_processes.reliability.k_of_n import k_of_n_av, k_of_n_rate
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
    flr=ceil-1
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
    flr=ceil-1
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
    if l>1:
        numr1 = h*(l_pod_k_of_n_av(l-1,n-ceil,k,p,q,ceil)-\
                    l_pod_k_of_n_av(l-1,n-ceil,k,p,q))
    else:
        numr1 = k_of_n_av(k,n,p)
    #Next, the joe pods
    if l>1:
        numr2 = j*(l_pod_k_of_n_av(l-1,n-flr,k,p,q,flr)-\
                l_pod_k_of_n_av(l-1,n-flr,k,p,q))
    else:
        numr2 = 0
    #Next, the hero machines
    if l>1:
        ## Condition on the pod of the hero machine in qn.
        hero_mc_up = q*l_pod_k_of_n_av(l,n-1,k-1,p,q) \
        #        + (1-q)*l_pod_k_of_n_av(l-1,n-ceil,k,p,q)
        hero_mc_down = q*l_pod_k_of_n_av(l,n-1,k,p,q) \
        #        + (1-q)*l_pod_k_of_n_av(l-1,n-ceil,k,p,q)
        numr3 = n_h*(hero_mc_up-hero_mc_down)
    else:
        numr3 = n_h*q*(k_of_n_av(k-1,n-1,p)-k_of_n_av(k,n-1,p))
    #Next, the joe machines
    ## First, availability when this Joe M/C is up.
    # Note that the second part is not needed since 
    # it cancels out with the next term.
    if l>1:
        joe_mc_up = q*l_pod_k_of_n_av(l-1,n-flr,k-1,p,q,flr-1)
        #           +(1-q)*l_pod_k_of_n_av(l-1,n-flr,k,p,q)
        #
        ## Next, the availability when this Joe M/C is down.
        joe_mc_down = q*l_pod_k_of_n_av(l-1,n-flr,k,p,q,flr-1)
        #            +(1-q)*l_pod_k_of_n_av(l-1,n-flr,k,p,q)
        #
        numr4 = n_j*(joe_mc_up-joe_mc_down)
    else:
        ## If there is only one pod, there will be no Joe machines.
        numr4=0
    ## Now the final numerator:
    numr=q*lmb_pod*(numr1+numr2)+p*lmb_mc*(numr3+numr4) 
    ## Now the denominator, which is just reliability
    denom = l_pod_k_of_n_av(l,n,k,p,q)
    return numr/denom

#hero_mc_up = q*l_pod_k_of_n_av(l-1,n-1,k-1,p,q) \
#                + (1-q)*l_pod_k_of_n_av(l-1,n-1,k,p,q)
#hero_mc_down = l_pod_k_of_n_av(l-1,n-1,k,p,q)

def l_pod_k_of_n_av_legacy(n=7, l=3, k=4):
    """
    This has an exponential complexity with l.
    """
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
def l_pod_k_of_n_sim(l,n,k,lmb_mc,mu_mc,lmb_pod,mu_pod,
                    durtn=9000,sim_durtn=1e4):
    ceil=np.ceil(n/l)
    flr=ceil-1
    h=int(n-l*flr)
    j=int(l*ceil-n)
    nodes = []
    for _ in range(h):
        pod_nodes = sim_one_pod(ceil,lmb_mc,mu_mc,lmb_pod,mu_pod,
                        durtn,sim_durtn)
        nodes.append(pod_nodes)
    for _ in range(j):
        pod_nodes = sim_one_pod(flr,lmb_mc,mu_mc,lmb_pod,mu_pod,
                        durtn, sim_durtn)
        nodes.append(pod_nodes)
    dat = pd.concat(nodes)
    dat = dat.sort_values(by=['start'])
    ds,ts=num_down_w_times_v1(np.array(dat.start),\
                    np.array(dat.end),np.array(dat.down))
    res_df=pd.DataFrame({'start':ts[:len(ts)-1],'end':ts[1:],'down':ds})
    durtns = res_df.end-res_df.start
    res_df = res_df[durtns>0]
    downs = res_df[res_df.down>=n-k+1]
    down_durtn = sum(downs.end-downs.start)
    up_durtn = durtn-down_durtn
    av = up_durtn/durtn
    downs = interrupts(res_df.down,n-k+1)
    air = downs/up_durtn
    return av, air


def sim_one_pod(machines,lmb_mc,mu_mc,lmb_pod,mu_pod,
                durtn=9000,sim_durtn=1e4):
    if durtn>sim_durtn:
        print("taking entire duration")
        durtn = sim_durtn
    nodes = []
    cut_pt = sim_durtn-durtn
    for _ in range(int(machines)):
        node1=birth_death_gen(lmb_mc,mu_mc,sim_durtn)
        node1 = node1[(node1.start>cut_pt) & (node1.state=="down")]
        nodes.append(node1)
    dat = pd.concat(nodes)
    dat = dat.sort_values(by=['start'])
    dat1 = complete_intervals(dat)[['start','end','down']]
    ##Now simulate the pod..
    pod1=birth_death_gen(lmb_pod,mu_pod,sim_durtn)
    pod1 = pod1[(pod1.start>cut_pt) & (pod1.state=="down")]
    pod1["down"] = machines
    pod1=pod1[['start','end','down']]
    pod_dat = pd.concat([dat1,pod1])
    pod_dat=pod_dat.sort_values(by=['start'])
    ds,ts=num_down_w_times_v1(np.array(pod_dat.start),\
            np.array(pod_dat.end),np.array(pod_dat.down))
    ds[ds>machines]=machines
    res_df=pd.DataFrame({'start':ts[:len(ts)-1],'end':ts[1:],'down':ds})
    durtns = res_df.end-res_df.start
    res_df = res_df[durtns>0]
    return res_df


def interrupts(downs,k):
    prev_dwn = 0
    interrupts=0
    for dwn in np.array(downs):
        if prev_dwn<k and dwn>=k:
            interrupts+=1
        prev_dwn=dwn
    return interrupts


def tst():
    av_sim=l_pod_k_of_n_sim(1,3,2,1,2,1,2)
    av_real=l_pod_k_of_n_av(1,3,2,0.66666,0.666666)

    ### with one pod that's almost always available, we get a k-of-n system.
    ## Note that the rate does not depend on mu of pods (last argument).
    l_pod_k_of_n_rate(1,3,2,1,2,.01,1)
    k_of_n_rate(2,3,1,2)


    l=3;n=3;k=2;p=2/(1+2);q=2/(1+2)
    diff = (l_pod_k_of_n_sim(3,3,2,1,2,1,2)[0] -l_pod_k_of_n_av(l,n,k,p,q))\
                /l_pod_k_of_n_av(l,n,k,p,q)

    l_pod_k_of_n_sim(1,2,2,1,2,1,2)
    .6666667*k_of_n_av(2,2,.6666666)

    l_pod_k_of_n_sim(1,3,2,1,2,1,2)[1]
    l_pod_k_of_n_rate(1,3,2,1,2,1,2)


    ### with n==l pods, we get k-of-n system.
    av, rate=l_pod_k_of_n_sim(3,3,2,1,2,1,2)
    av1=l_pod_k_of_n_av(3,3,2,.6666666,.66666666)
    rate1=l_pod_k_of_n_rate(3,3,2,1,2,1,2)
    rate2=k_of_n_rate(2,3,2,1.6)


    l_pod_k_of_n_sim(2,3,2,1,2,.0001,2)
    l_pod_k_of_n_rate(2,3,2,1,2,.0001,2)
    l_pod_k_of_n_av(3,3,2,.6666666,2/2.0001)
    k_of_n_rate(2,3,1,2)

    l_pod_k_of_n_sim(2,7,5,1,2,1,2)
    l_pod_k_of_n_rate(2,7,5,1,2,1,2)

    l_pod_k_of_n_rate(2,7,4,1,2,.00001,2)
    k_of_n_rate(4,7,1,2)

    l_pod_k_of_n_rate(3,9,4,.00001,2,1,2)
    k_of_n_rate(2,3,1,2)

    l_pod_k_of_n_rate(3,8,4,.00001,2,1,2)
    k_of_n_rate(2,3,1,2)

