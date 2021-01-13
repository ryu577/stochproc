import numpy as np

# Let’s say we have 3 nodes and observe all three for a month.
# Node 1: UP for 20 days; then goes to OFR (causing a fault and goes out of production); 
#   then comes back to production on day 25 (start), runs fine until going back to OFR on 
#   day 29 (start; so ran for 4 days) and stays out until end of month.
# Node 2: UP for 10 days; has a hardware fault (non-OFR) and comes back up immediately and stays 
#   UP for the remaining 20 days.
# Node 3: UP for the whole 30 days.

## Now, let's simulate a customer that comes and runs VMs on this kind of data.
def rate_sim():
    total_failure_time = 0

    for i in range(10000):
        time_to_failure = 0
        failed = False
        while not failed:
            # Customer enters randomly sometime in the month.
            customer_enters_at_time = np.random.uniform()*30
            # If he enters between days 24 and 29, he is randomly allocated to node 1 or 3
            # Otherwise, he is randomly allocated to node 1, 2 or 3.
            if (20<customer_enters_at_time and customer_enters_at_time<25) or \
                    (29<customer_enters_at_time and customer_enters_at_time<30):
                node_allotted = np.random.choice([2,3])
            else:
                node_allotted = np.random.choice(3)+1
            if node_allotted==3:
                time_to_failure += (30-customer_enters_at_time)
            elif node_allotted==2:
                if customer_enters_at_time>10:
                    time_to_failure += (30-customer_enters_at_time)
                else:
                    time_to_failure += (10-customer_enters_at_time)
                    failed = True
            elif node_allotted==1:
                if customer_enters_at_time<20:
                    time_to_failure += (20-customer_enters_at_time)
                    failed=True
                else:
                    time_to_failure += (29-customer_enters_at_time)
                    failed = True
        total_failure_time += time_to_failure

    print("MTBF:" + str(total_failure_time/10000))


