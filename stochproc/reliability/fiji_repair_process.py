import numpy as np
from algorith.heap.heap import Heap

#### Why is this a function of total time we run the simulation for?

### Inputs to the program.
## The number of machines in the site.
n=20
sh_durtn = 80
sh_air=3
mtbf = 365*1440*100/sh_air

tech_arrival = np.inf; prev_tech_arrival=0
down_heap = Heap()
for i in range(n):
    down_heap.push(np.random.exponential(mtbf))

to_repair=Heap()

curr_t=0; downs=0
down_inter=0; up_inter=0; tot_inter=0

while curr_t < mtbf*200:
    while down_heap.peek()<tech_arrival:
        t=down_heap.pop()
        downs+=1; to_repair.push(t)
        curr_t=t
        if downs==1:
            tech_arrival=min(tech_arrival,t+14*1400)
        elif downs==2:
            tech_arrival=min(tech_arrival,t+2*1440)
    curr_t=tech_arrival
    for _ in range(downs):
        down_heap.push(curr_t+np.random.exponential(mtbf))
    tot_inter = n*curr_t
    down_inter+=sh_durtn
    # First time stamp would have been service healed,
    # which is already accounted for. So pop it.
    to_repair.pop()
    while len(to_repair.h_arr)>0:
        t1 = to_repair.pop()
        down_inter+=(curr_t-t1)        
    prev_tech_arrival = tech_arrival
    to_repair = Heap()
    tech_arrival=np.inf
    downs=0

print(1-down_inter/tot_inter)

