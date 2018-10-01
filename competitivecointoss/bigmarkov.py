import numpy as np


class State:
    def __init__(self, n1, n2):
        self.n1 = n1 # n1 in [0, 3]
        self.n2 = n2 # n2 in [0,2]
        self.id = n2*4 + n1

    def __str__(self):
        return str(self.n1) + "," + str(self.n2)


class Transition:
    def __init__(self, source, desti, prob):
        self.source = source
        self.desti = desti
        self.prob = prob

    def __str__(self):
        return "souce:" + str(self.source) + ", desti:" + str(self.desti) + ", prob:" + str(self.prob)


def getTransitions(source):
    n1 = source.n1
    n2 = source.n2

    if n1 == 3 and n2 == 2:
        return [
            Transition(source,State(0, 0), 1),
        ]

    if n1==3:
        return [
        Transition(source, source, 1),
    ]

    if n2==2:
        return [
        Transition(source, source, 1),
    ]

    return [
        Transition(source, State(n1+1, n2+1), 0.25),
        Transition(source, State(n1+1, 0), 0.25),
        Transition(source, State(0, n2+1), 0.25),
        Transition(source, State(0, 0), 0.25),
    ]


totalTrans = []
for n2 in range(0, 3):
    for n1 in range(0, 4):
        trans = getTransitions(State(n1, n2))
        totalTrans = totalTrans + trans

m = np.zeros([12, 12])
for t in totalTrans:
    m[t.desti.id, t.source.id] = t.prob

print(m)

start = np.zeros([12])
start[0] =1

for i in range(0, 10000):
    if i%100 == 0:
        # print(start)
        print("step {}: win prob for state (3,0) is {}, win prob for state (3,1) is {}, win prob for state (3,2) is {}".format(i, start[3], start[7], start[11]))
    start = np.dot(m, start)
 


