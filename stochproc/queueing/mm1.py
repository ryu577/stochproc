import numpy as np
import matplotlib.pyplot as plt


def mm1_queue(lmb=1, mu=3):
    """
    Simulates the number of customers in the system for 
    an M/M/1 queue and compares with the closed form.
    """
    arrival_intervals = np.random.exponential(scale=1/lmb,size=2000)
    arrival_times = np.cumsum(arrival_intervals)
    n = 0
    customers_in_system = []
    state_change_times = []
    for i in range(1,len(arrival_times)):
        customers_in_system.append(n)
        state_change_times.append(arrival_times[i-1])
        n += 1
        prev_activity_time = arrival_times[i-1]
        while n > 0:
            next_service_time = np.random.exponential(scale=1/mu)
            prev_activity_time += next_service_time
            if prev_activity_time > arrival_times[i]:
                break
            else:
                state_change_times.append(prev_activity_time)
                customers_in_system.append(n)
                n -= 1    
    intervals = np.diff(state_change_times)
    state_change_intervals = np.concatenate(([arrival_times[0]],intervals),axis=0)
    avgs = np.cumsum(state_change_intervals*customers_in_system)/state_change_times
    plt.plot(state_change_times, avgs)
    plt.axhline(lmb/(mu-lmb))
    plt.show()



