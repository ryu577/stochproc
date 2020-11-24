import numpy as np
from algorith.heap.heap import Heap

## No reserve..
## Reserve one.

#### Why is this a function of total time we run the simulation for?

### Inputs to the program.
## The number of machines in the site.
n=8
sh_durtn = 30
sh_air=7
# Mean time between failures for reboots due to service healing.
mtbf = 365*1440*100/sh_air
# Do we have a one node buffer or not.
no_buffer=False

tech_arrival = np.inf; prev_tech_arrival=0
down_heap = Heap()
for i in range(n):
    down_heap.push(np.random.exponential(mtbf))

to_repair=Heap()

curr_t=0; downs=0
down_inter=0; up_inter=0; tot_inter=0

## Simulate until some large time.
while curr_t < mtbf*100:
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
    if no_buffer:
        #If we have one node in the buffer, the first failure will be service healable.
        down_inter+=sh_durtn
        # First time stamp would have been service healed,
        # which is already accounted for. So pop it.
        to_repair.pop()
    # For the remaining failures, nodes have been down since the times in the heap.
    while len(to_repair.h_arr)>0:
        t1 = to_repair.pop()
        down_inter+=(curr_t-t1)        
    prev_tech_arrival = tech_arrival
    to_repair = Heap()
    tech_arrival=np.inf
    downs=0

print(1-down_inter/tot_inter)

