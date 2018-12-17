import numpy as np

m = np.array([[0,.5,0,.5,0],
              [.5,0,.5,0,0],
              [0,.5,0,0,.5],
              [0,0,0,1.0,0],
              [0,0,0,0,1.0]])

mm = np.linalg.matrix_power(m,30)


def walk_matrix(bottom, target, p=0.5):
    state_size = bottom+target+1
    transient_size = state_size-2
    absorbing_size = 2
    m = np.zeros((state_size, state_size))
    q = np.zeros((transient_size, transient_size))
    r = np.zeros((transient_size, absorbing_size))
    m[0,0] = 1
    m[state_size-1, state_size-1] = 1
    r[0,0]=1-p; m[0,1] = 1-p
    r[transient_size-1,1] = p; m[state_size-2, state_size-1] = p
    for i in range(transient_size-1):
        q[i,i+1] = p; m[i+1,i+2] = p
        q[i+1,i] = 1-p; m[i+2,i+1] = 1-p
    probs = np.linalg.solve(np.eye(transient_size)-q,r)[bottom-1]
    return probs, m


def simulate_gambler(k1=2, k2=3):
    # 0 for g1, 1 for g2 and 2 for draw.
    results = np.zeros(3)
    for i in range(1000):
        g1 = g2 = 0
        for j in range(4000):
            if np.random.uniform() > 0.5:
                g1 += 1
            else:
                g1 -= 1
            if np.random.uniform() > 0.5:
                g2 += 1
            else:
                g2 -= 1
            if g1 == 2 and g2 == 3:
                results[2] += 1
                break
            elif g1 == 2:
                results[0] += 1
                break
            elif g2 == 3:
                results[1] += 1
                break
            elif g1 < -300 and g2 < -300:
                results[0] += 1
                break
    return results


