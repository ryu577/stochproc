import numpy as np


def play():
    victor = 2
    pl1 = 0
    pl2 = 0
    while victor == 2:
        toss1 = (np.random.uniform() < 0.5)
        toss2 = (np.random.uniform() < 0.5)
        if toss1:
            pl1 += 1
            pl1 = min(pl1, 2)
        else:
            pl1 = 0
        if toss2:
            pl2 += 1
            pl2 = min(pl2, 3)
        else:
            pl2 = 0
        if pl1==2 and pl2==3:
            victor = 0

        if pl1 == 2 and pl2 < 3:
            victor = 0

        if pl2 == 3 and pl1 < 2:
            victor = 1
            # elif pl1 == 2 and pl2 == 3:
            #    pl1 = pl2 = 0
    return victor


def get_winner_prob(N = 1000):
    cnt = 0
    for i in range(N):
        cnt += play()
    return cnt/N


def run_multiple_sim():
    prob = []
    for l in range(1000):
        prob.append(get_winner_prob())
    print(np.average(prob))
    print(np.std(prob))


def play1(cur_toss = 'x'):
    curr_bool = (np.random.uniform()<0.5)
    if curr_bool:
        nxt_toss = 'h'
    else:
        nxt_toss = 't'
    seq = cur_toss + nxt_toss
    return seq, nxt_toss


def get_loser_state_prob():
    set1 = {'hh':0,'ht':1,'th':2,'tt':3}
    a = np.zeros(4)
    for i in range(1000000):
        cur1 = cur2 = 'x'
        while True:
            seq1, cur1 = play1(cur1)
            seq2, cur2 = play1(cur2)
            #print(seq1 + "\t" + seq2)
            if seq1 == 'hh':
                a[set1[seq2]] += 1
                break
            elif seq2 == 'hh':
                a[set1[seq1]] += 1
                break
    print(str(a/1000000))


