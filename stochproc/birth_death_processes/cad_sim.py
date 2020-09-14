import numpy as np
from stochproc.birth_death_processes.birth_death_gen import birth_death_gen

def birth_death_availability():
    availab = 0
    nsim=10000
    lmb=10.0
    mu=2.0
    t=1000    
    for _ in range(nsim):
        up=True
        cum_t=0
        while cum_t<t:
            state = "up" if up else "down"
            t1 = np.random.exponential(1/lmb) \
                    if up else np.random.exponential(1/mu)
            cum_t += t1
            up=not up
        if state == "up":
            availab+=1
    return availab/nsim


class IntervalData(object):
    def __init__(self,lmb_off,lmb_on,lmb_down,lmb_up,t_stop):
        """
        lmb_off: Rate at which customer turns VM off when its ON.
        lmb_on: Rate at which customer turns VM on when its OFF.
        lmb_down: The rate at which VM goes down when running. This
                  is the "AIR" in Azure.
        lmb_up: Rate at which VM comes back up when down.
        t_stop: The time at which we stop simulating.
        """
        self.lmb_off=lmb_off
        self.lmb_on=lmb_on
        self.lmb_down=lmb_down
        self.lmb_up=lmb_up
        self.t_stop=t_stop

    def data_gen(self):
        up_down_dat = birth_death_gen(self.lmb_down, self.lmb_up, self.t_stop)
        start_stop_dat = birth_death_gen(self.lmb_off, self.lmb_on, self.t_stop)
        start_stop_dat.loc[start_stop_dat.state=="up","state"]="running"
        start_stop_dat.loc[start_stop_dat.state=="down","state"]="stopped"
        start_stop_dat.loc[0,"start"]=-1e-3
        self.interval_dat = combine_intervals(start_stop_dat,up_down_dat)


def combine_intervals(start_stop_dat,up_down_dat):
    uniond = pd.concat((start_stop_dat[["start","state"]],\
                    up_down_dat[["start","state"]]))
    uniond = uniond.sort_values(by=["start"])
    uniond["end"] = np.concatenate((uniond.start[1:],[1000]))
    uniond = uniond.loc[uniond.state != "stopped"]
    #Remove the dummy VM-start
    uniond = uniond[1:]
    states = []
    state = True
    for st in uniond.state:
        if st == "stopped":
            state = False
        elif st == "started":
            state = True
        states.append(state)
    uniond = uniond[states]
    return uniond


def tst_gen_cad_data():
    ida = IntervalData(.9,9.0,.01,100,1000)
    ida.data_gen()
    cad_dat = ida.interval_dat

