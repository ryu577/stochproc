import numpy as np
from scipy.stats import weibull_min


def wait_time(c=.295):
    """
    Fred then visits Blunderville, where the times between buses are also 
    10 minutes on average, and independent. Yet to his dismay, he finds 
    that on average he has to wait more than 1 hour for the next bus when 
    he arrives at the bus stop! How is it possible that the average 
    Fred-to-bus time is greater than the average bus-to-bus time even 
    though Fred arrives at some time between two bus arrivals? Explain 
    this intuitively, and construct a specific discrete distribution 
    for the times between buses showing that this is possible.
    """
    time = 0
    arrival = 11543.674
    mu = weibull_min.mean(c)
    while time < arrival:
        time += 10/mu*weibull_min.rvs(c)
    return time - arrival


def avg_arrival(c=.295, n=1000):
    print("Mean: " + str(weibull_min.stats(c, moments='m')))
    sum_t = 0
    for _ in range(n):
        sum_t += wait_time(c)
    return sum_t/n


if __name__ == "__main__":
    avg_arrival()
