import numpy as np
from algorith.data_structs.heap.heap import Heap


class Trnstn():
    def __init__(self, abs_t, rel_t, up, ix):
        self.abs_t = abs_t
        self.rel_t = rel_t
        self.up = up
        self.ix = ix

    def __eq__(self, other):
        return (self.abs_t == other.abs_t)

    def __ne__(self, other):
        return self.abs_t != other.abs_t

    def __lt__(self, other):
        return (self.abs_t < other.abs_t)

    def __le__(self, other):
        return (self.abs_t <= other.abs_t)

    def __gt__(self, other):
        return (self.abs_t > other.abs_t)

    def __ge__(self, other):
        return (self.abs_t >= other.abs_t)


def sim_series():
    dwn_evnts = 0
    up_t = 0
    for ii in range(10):
        sim_until = 1e4
        rates = np.array([[1, 1.1, 1.2], [2, 2, 2]])
        n = len(rates[0])
        hp = Heap()
        prev_t = 0
        j = 0
        for i in rates[0]:
            t_rand = np.random.exponential(1/i)
            trnstn = Trnstn(t_rand, t_rand, 0, j)
            j += 1
            hp.push(trnstn)

        t = 0
        ups = n
        dwns = 0
        t_up = 0
        t_dwn = 0

        totl_t = 0

        while t < sim_until:
            curr_trnstn = hp.pop()
            t = curr_trnstn.abs_t
            nxt_state = (curr_trnstn.up + 1) % 2
            lmbd = rates[nxt_state][curr_trnstn.ix]
            delt = np.random.exponential(1/lmbd)
            nxt_trnstn = Trnstn(t+delt, delt, nxt_state, curr_trnstn.ix)
            hp.push(nxt_trnstn)
            delt_1 = (curr_trnstn.abs_t-prev_t)
            t_up += ups*delt_1
            t_dwn += (n-ups)*delt_1
            # Just as a sanity check, this should equal t afer loop.
            totl_t += (t-prev_t)
            # if all n components are UP and you're transitioning,
            # you can only be transitioning DOWN (series system).
            if ups == n:
                up_t += (t-prev_t)
            if curr_trnstn.up == 1:
                ups += 1
            else:
                if ups == n:
                    dwn_evnts += 1
                ups -= 1
            if t < sim_until:
                prev_t = t

        if ups == n:
            up_t += (sim_until - prev_t)

    # After loop, t==totl_t > sim_until > prev_t
    print(dwn_evnts/up_t)
