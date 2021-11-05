import numpy as np
import pandas as pd
from algorith.arrays.birth_death.cad_as_violations import complete_intervals, num_down


def birth_death_gen(lmb=1.0, mu=2.0, t=10000):
    """
    Simulates from a birth and death process
    like VM life, where the VM suffers interruptions at the
    rate lmb when its running and recovery happens at the
    rate mu once its down. The intervals start at 0 and end
    at time t.
    """
    up = True
    ts = []
    cum_t = 0
    states = []
    starts=[]; ends=[]
    while cum_t<t:
        starts.append(cum_t)
        rate = lmb if up else mu
        state = "up" if up else "down"
        t1 = np.random.exponential(1/rate)
        ts.append(t1)
        states.append(state)
        cum_t += t1
        up=not up
        ends.append(min(cum_t,t))
    dat = pd.DataFrame({"durtn":ts,"state":states,"start":starts,"end":ends})
    return dat
