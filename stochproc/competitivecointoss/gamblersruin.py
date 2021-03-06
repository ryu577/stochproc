import numpy as np
from scipy.special import comb, gamma
import scipy as sp

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
    #probs = sp.linalg.solve_banded((1,1),np.eye(transient_size)-q,r)[bottom-1]
    return probs, m


def simulate_gambler(k1=2, k2=3):
    # 0 for g1, 1 for g2 and 2 for draw.
    results = np.zeros(3)
    for i in range(10000):
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


def binom_term(n, k, p):
    ans = 0
    for i in range(k):
        ans += np.log((n-i)) + np.log(p) -np.log(k-i)
    for i in range(n-k):
        ans += np.log(n-k-i)+np.log(1-p) - np.log(n-k-i)
    return np.exp(ans)


def win_sequence(k, p=0.5, size=1000):
    seq = np.zeros(size)
    for l in range(int((size-k)/2)):
        #seq[k+2*l] = k/(k+l)*comb(k+2*l-1,l)*p**(k+l)*(1-p)**l
        seq[k+2*l] = k/(k+l)*binom_term(k+2*l-1,l,p)*p**k*(1-p)**(1-k)
    return seq


def gambler_race(k1=2, k2=3, size=1000):
    a1 = win_sequence(k2,0.5,size)
    cum_a1 = 0; cum_a2 = 0
    a2 = win_sequence(k1,0.5,size)
    ans1 = 0; ans2 = 0
    for n in range(len(a1)):
        cum_a1 += a1[n]
        cum_a2 += a2[n]
        ans1 += (1-cum_a1)*a2[n]
        ans2 += (1-cum_a2)*a1[n]
    residue = 1 - ans1 - ans2
    return ans1, residue, ans1+residue/2


def gambler_race_v2():
    a_t = np.array([(comb(2*t+1,t)-comb(2*t+1,t-1))/2**(2*t+2) for t in range(500)])
    b_t = np.array([(comb(2*t+2,t)-comb(2*t+2,t-1))/2**(2*t+3) for t in range(500)])
    b_sum = 1-np.concatenate(([0],np.cumsum(b_t)))
    return sum(a_t*b_sum[:500])


def gamblers_race_closed_form():
    seq1 = 1 - np.array([(3*t+4)*gamma(t+1.5)/gamma(t+3)/np.sqrt(np.pi) for t in range(20)])
    seq2 = 1 - np.array([(3*t+7)*gamma(t+2.5)/gamma(t+4)/np.sqrt(np.pi) for t in range(20)])


def draw_recurrence_relation():
    k = 3
    d_k_1 = np.array([16,-5])
    d_k_2 = np.array([4,-1])
    for k1 in range(15):
        d_k = (2*(6*k**2-12*k+5)*d_k_1 - (k-2)*(2*k-1)*d_k_2 - np.array([8,0]))/k/(2*k-3)
        print(d_k[0]/d_k[1])
        d_k_2 = d_k_1
        d_k_1 = d_k    
        k+=1


"""
Mathematica code:
a[t_] :=
  a[t] = (Binomial[2 t + 1, t] - Binomial[2 t + 1, t - 1])/2^(2 + 2 t);
b[t_] :=
  b[t] = (Binomial[2 t + 2, t] - Binomial[2 t + 2, t - 1])/2^(3 + 2 t);
{
 {Sum[ a[t] (1 - Sum[b[j], {j, 0, t - 1}]), {t, 0, \[Infinity]}]}, {\[Placeholder]}
}
"""
