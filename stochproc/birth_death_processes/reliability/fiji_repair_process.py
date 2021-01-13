import numpy as np
from algorith.heap.heap import Heap


### Inputs to the program.
## The number of machines in the site.


def fiji_repair_availability(n=8,sh_durtn=30,sh_air=7,buffer=True,sim_epochs=100):
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

    ## Simulate until some large time.
    while curr_t < mtbf*sim_epochs:
        while down_heap.peek()<tech_arrival:
            t=down_heap.pop()
            downs+=1; to_repair.push(t)
            curr_t=t
            if downs==1:
                # One node goes down, tech is scheduled to arrive in 14 days.
                tech_arrival=min(tech_arrival,t+14*1400)
            elif downs==2:
                # If two nodes fail, tech arrival is updated to one day from now.
                tech_arrival=min(tech_arrival,t+1*1440)
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


if __name__=="__main__":
    n=8
    sh_durtn = 30
    sh_air=7
    av=fiji_repair_availability(8,30,sh_air=4,sim_epochs=5000)
    print(av)

