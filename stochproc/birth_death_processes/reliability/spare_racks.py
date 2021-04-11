import numpy as np
from algorith.data_structs.heap.heap import Heap


class Rack():
    def __init__(self, ix, is_spare, lmb, mu, av=1):
        self.ix = ix
        self.is_spare = is_spare
        self.lmb = lmb
        self.mu = mu
        self.av = av


class Evnt():
    def __init__(self, change_t, type='rack_state_ch', rack=None):
        self.change_t = change_t
        self.type = type
        if type == 'rack_state_ch':
            self.rack = rack

    def __eq__(self, other):
        return (self.change_t == other.change_t)

    def __ne__(self, other):
        return self.change_t != other.change_t

    def __lt__(self, other):
        return (self.change_t < other.change_t)

    def __le__(self, other):
        return (self.change_t <= other.change_t)

    def __gt__(self, other):
        return (self.change_t > other.change_t)

    def __ge__(self, other):
        return (self.change_t >= other.change_t)


def sim_rack_proc(num_racks=10, have_spare=True):
    lmbds = np.ones(num_racks)
    mus = np.ones(num_racks)*2
    if have_spare:
        spare_ixs = {len(lmbds)-1}
    else:
        spare_ixs = set()
    racks = []
    end_t = 100000
    terminal_evnt = Evnt(end_t, "terminal", None)
    hp = Heap()
    hp.push(terminal_evnt)
    ups = 0
    for i in range(len(lmbds)):
        if i not in spare_ixs:
            rck = Rack(i, 0, lmbds[i], mus[i], 1)
            t_dwn = np.random.exponential(1/lmbds[i])
            ev = Evnt(t_dwn, rack=rck)
            hp.push(ev)
            ups += 1
        else:
            rck = Rack(i, 1, lmbds[i], mus[i], 1)
        racks.append(rck)

    no_sh_dwns = 0
    prev_t = 0
    total_up_t = 0
    while True:
        ev = hp.pop()
        curr_t = ev.change_t
        total_up_t += (curr_t - prev_t) * ups
        if ev.type == "terminal":
            break
        elif ev.type == 'rack_state_ch':
            # UP going DOWN
            if ev.rack.av == 1:
                ev.rack.av = 0
                nx_t = np.random.exponential(1/ev.rack.mu)
                ev.change_t = curr_t + nx_t
                if len(spare_ixs) == 1:
                    ev.rack.is_spare = 1
                    spare_ix = spare_ixs.pop()
                    rck = racks[spare_ix]
                    rck.is_spare = 0
                    nx_t = np.random.exponential(1/rck.lmb)
                    ev1 = Evnt(curr_t + nx_t, rack=rck)
                    hp.push(ev1)
                elif len(spare_ixs) == 0:
                    ups -= 1
                    no_sh_dwns += 1
                hp.push(ev)
            # DOWN going UP
            else:
                ev.rack.av = 1
                # For now we assume that if the spare comes
                # back and there are still racks down, their
                # VMs are not service healed to this freshly
                # available rack.
                if ev.rack.is_spare == 1:
                    spare_ixs.add(ev.rack.ix)
                else:
                    ups += 1
                    nx_t = np.random.exponential(1/ev.rack.lmb)
                    ev.change_t = curr_t + nx_t
                    hp.push(ev)
        prev_t = curr_t
    return no_sh_dwns, total_up_t, no_sh_dwns/total_up_t

# Also see: stochproc\tests\scratch.py
