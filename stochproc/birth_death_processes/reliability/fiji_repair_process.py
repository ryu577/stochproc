import numpy as np
from algorith.data_structs.heap.heap import Heap
from stochproc.birth_death_processes.reliability.series_parallel import series

### Inputs to the program.
## The number of machines in the site.


def fiji_repair_availability(n=8, sh_durtn=30, sh_air=7, buffer=True,
                             sim_epochs=100, repair_sla_days=1):
    # Mean time between failures for reboots due to service healing.
    mtbf = 365*1440*100/sh_air
    # Do we have a one node buffer or not.

    azure_sh_av = mtbf/(mtbf+sh_durtn)
    print("Availability contribution in Azure due to SH: " + str(azure_sh_av))
    tech_arrival = np.inf; prev_tech_arrival=0
    down_heap = Heap()
    for _ in range(n):
        down_heap.push(np.random.exponential(mtbf))

    to_repair=Heap()

    curr_t=0; downs=0
    down_inter=0; up_inter=0; tot_inter=0

    # Simulate until some large time.
    while curr_t < mtbf*sim_epochs:
        while down_heap.peek()<tech_arrival:
            t=down_heap.pop()
            downs+=1; to_repair.push(t)
            curr_t=t
            if downs==1:
                # One node goes down, tech is scheduled to arrive in 14 days.
                tech_arrival=min(tech_arrival, t+14*1400)
            elif downs==2:
                # If two nodes fail, tech arrival is updated to one day from now.
                tech_arrival=min(tech_arrival, t+repair_sla_days*1440)
        curr_t=tech_arrival
        for _ in range(downs):
            down_heap.push(curr_t+np.random.exponential(mtbf))
        tot_inter = n*curr_t
        up_slb_nodes=3; up_hosting_nodes=n
        if buffer:
            #If we have one node in the buffer, 
            # the first failure will be service healable.
            # But was it a non SLB node? If not, there is no impact.
            if np.random.uniform()<up_hosting_nodes/(up_slb_nodes+up_hosting_nodes):
                down_inter+=sh_durtn
                up_hosting_nodes-=1
            else:
                up_slb_nodes-=1
            # First time stamp would have been service healed,
            # which is already accounted for. So pop it.
            to_repair.pop()
        # For the remaining failures, nodes have
        # been down since the times in the heap.
        while len(to_repair.h_arr)>0:
            t1 = to_repair.pop()
            if np.random.uniform()<up_hosting_nodes/(up_slb_nodes+up_hosting_nodes):
                down_inter+=(curr_t-t1)
                up_hosting_nodes-=1
            else:
                up_slb_nodes-=1
                # If we're left with less than 2 SLB nodes, 
                # everything was down before the tech arrived 
                # since the second one went down.
                if up_slb_nodes<2:
                    down_inter+=(curr_t-t1)*n
                    to_repair=Heap()
        prev_tech_arrival = tech_arrival
        to_repair = Heap()
        tech_arrival=np.inf
        downs=0
    return 1-down_inter/tot_inter


def overall_metrics(sla_days=1):
    """
    What impact does SLA for SH policy have on overall availability?
    """
    lmb_sh = 4.22
    a_sh = fiji_repair_availability(sh_air=lmb_sh,
                                    sim_epochs=5000, n=5,
                                    repair_sla_days=
                                    sla_days)
    a_sh_1day = fiji_repair_availability(sh_air=lmb_sh,
                                    sim_epochs=5000, n=5,
                                    repair_sla_days=1)
    mu_sh = lmb_sh*a_sh/(1-a_sh)
    a_t = 0.9997831556492641
    lmb_t = 39.54458631633028
    a_nosh = a_t/a_sh_1day
    lmb_nosh = lmb_t - lmb_sh
    mu_nosh = lmb_nosh*a_nosh/(1-a_nosh)
    lmb_new, mu_new, a_new = series(np.array([lmb_nosh, lmb_sh]), np.array([mu_nosh, mu_sh]))
    return lmb_new, mu_new, a_new, 36500*1440/mu_sh


def fiji_longdown_rate(theta=0.2, days=14):
    period = 1000000
    n_sim = 1000
    s_evnts = 0
    for i in range(n_sim):
        trntns = []
        tech_arrival = np.inf
        num_dwn = 0
        t = 0
        evnts = 0
        lm = np.random.gamma(1.4*theta, 1/theta)*7/36500
        # lm = 5/(36500/7)
        while t < period:
            t_del = np.random.exponential(1/lm)
            t += t_del
            if t > tech_arrival:
                trntns.append([tech_arrival, 0])
                trntns.append([t, 1])
                num_dwn = 1
                tech_arrival = t + days
            else:
                num_dwn += 1
                if num_dwn >= 2:
                    tech_arrival = min(tech_arrival, t+1)
                    evnts += 1
                trntns.append([t, num_dwn])
        s_evnts += evnts

    evnts_per_t_days = s_evnts/n_sim
    evnts_per_cent = evnts_per_t_days*36500/period
    print(evnts_per_cent)
    print(len(trntns))


def tst_neg_binom():
    period=1000
    evnts = 0
    t=0
    while t<period:
        theta = 0.5
        lm = np.random.gamma(5*theta,1/theta)
        #lm = 5
        t_del = np.random.exponential(1/lm)
        t+=t_del
        evnts+=1
    print(evnts)


if __name__=="__main__":
    n=8
    sh_durtn = 30
    sh_air=7
    av=fiji_repair_availability(8,30,sh_air=4,sim_epochs=5000)
    print(av)
