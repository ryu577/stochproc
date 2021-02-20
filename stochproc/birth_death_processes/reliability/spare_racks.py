# Rate of events defined as two or more failures.
import pandas as pd
from stochproc.birth_death_processes.birth_death_gen import birth_death_gen
from algorith.arrays.birth_death.cad_as_violations import complete_intervals


lmbds = np.ones(2)
mus = np.ones(2)*2
spares = 1
statuses = np.ones(len(lmbds))




def rack_failure_rate(racks):
    dat = pd.DataFrame(columns=["durtn", "state", "start", "end"])
    node_up_time = 0
    for k in range(racks):
        node1 = birth_death_gen(1, 2, 1e5)
        node1 = node1[(node1.start > 1e3)]
        node_d = node1[(node1.state == "down")]
        dat = pd.concat([dat, node_d])
        node_up_time += sum(node1[node1.state == "up"].durtn)
    dat = dat.sort_values(by=['start'])

    res = complete_intervals(dat)
    dnw_instances = 0
    prev_dwn = 0
    for i in res.down:
        if i > prev_dwn:
            dnw_instances += 1
        prev_dwn = i

    return dnw_instances/node_up_time
