## Courtesy Randolph Yao
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
        return "souce:" + str(self.source) + ", desti:" \
        + str(self.desti) + ", prob:" + str(self.prob)


def getTransitions(source):
    n1 = source.n1
    n2 = source.n2
    if n1 == 3 and n2 == 2:
        return [
            #Transition(source,State(0, 0), 1),
            Transition(source,source, 1),
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


def construct_big_matrix():
    m = np.zeros((12,12))
    state2index = {(0,0):0,(1,0):1,(2,0):2,(0,1):3,(1,1):4,(2,1):5,(0,2):6,(1,2):7,(2,2):8,(3,0):9,(3,1):10,(3,2):11}
    index2state = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1),(3,2)]
    for i in range(6):
        m[i,0] = 0.25
        m[i, state2index[(index2state[i][0]+1,0)]] = 0.25
        m[i, state2index[(0,index2state[i][1]+1)]] = 0.25
        m[i, state2index[(index2state[i][0]+1,index2state[i][1]+1)]] = 0.25
    m[6:,6:] = np.eye(6)
    q = m[:6,:6]
    r = m[:6,6:]
    u = np.linalg.solve(np.eye(6)-q, r)
    print("Probability you win:" + str(u[0,3]+u[0,4]))


def run_bigmarkov():    
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
    start[0] = 1
    for i in range(0, 10000):
        if i%100 == 0:
            # print(start)
            print("step " + str(i) + ": win prob for state (3,0) is " \
                + str(start[3]) +"," + \
                " win prob for state (3,1) is " +str(start[7]) + ","\
                 + " win prob for state (3,2) is " + str(start[11]))
        start = np.dot(m, start)


if __name__ == '__main__':
    run_bigmarkov()

