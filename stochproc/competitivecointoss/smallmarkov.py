import numpy as np
import operator as op
from functools import reduce
import abc
import matplotlib.pyplot as plt


class MarkovSequence:
    def __init__(self, trnsn_matrix):
        self.trnsn_matrix = trnsn_matrix
        self.seq_len = trnsn_matrix.shape[0]
        self.get_coef_matrix()

    def __mul__(self, other):
        return get_winner_prob(self, other)

    def __rmul__(self, other):
        return get_winner_prob(other, self)

    def get_coef_matrix(self):
        a = self.trnsn_matrix
        [eig_a, eig_vec_a] = np.linalg.eig(a)        
        self.eigs = eig_a
        use_eig_mat = True
        for i in range(1,len(eig_a)):
            if eig_a[i] == eig_a[i-1]:
                use_eig_mat = False
                break
        #exploit already calculated eigen vectors if possible.
        if use_eig_mat:
            eig_vec_a_inv = np.linalg.inv(eig_vec_a)
            # Taking the first row since we start in [1,0,0]
            first_row_e = np.array(eig_vec_a[0])[0]
            self.coef_matrix = np.dot(np.diag(first_row_e),\
                                       eig_vec_a_inv)
        else:
            self.coef_matrix = get_generalized_coefficients(\
                self.trnsn_matrix, self.eigs)
        self.ultimate_coefs = np.array(\
                                    self.coef_matrix.T[self.seq_len-1])[0]
        self.penultimate_coefs = np.array(\
                                    self.coef_matrix.T[self.seq_len-2])[0]


def get_winner_prob(win_seq, lose_seq, win_prob=0.5):
    a_c, a_e = np.copy(win_seq.penultimate_coefs), np.copy(win_seq.eigs)
    a_c = a_c/a_e
    #a_c = flip_seq(a_c, a_e)
    b_c, b_e = np.copy(lose_seq.ultimate_coefs), np.copy(lose_seq.eigs)
    b_c = flip_seq(b_c, b_e)
    ans = mult_seq(a_c, a_e, b_c, b_e)
    return ans*win_prob


def get_consecutive_heads_mat(numstates=3, p=0.5):
    trnsn_mat = np.matrix(np.zeros((numstates,numstates)))
    for i in range(trnsn_mat.shape[0]-1):
        trnsn_mat[i,0] = (1-p)
        trnsn_mat[i,i+1] = p
    trnsn_mat[numstates-1,numstates-1] = 1.0
    return trnsn_mat


def get_running_total_heads_mat(numstates=4, p=0.5):
    trnsn_mat = np.eye(numstates)*(1-p)
    for i in range(numstates-1):
        trnsn_mat[i,i+1] = p
    trnsn_mat[numstates-1,numstates-1]=1.0
    return np.matrix(trnsn_mat)


def get_coefs(a=np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]]), 
             ix=None):
    if ix is None:
        ix = a.shape[0]-1
    [eig_a, eig_vec_a] = np.linalg.eig(a)
    eig_vec_a_inv = np.linalg.inv(eig_vec_a)
    first_row_e = np.array(eig_vec_a[0])[0]
    last_col_e_inv = np.array(eig_vec_a_inv.T[ix])[0]
    coefs = first_row_e*last_col_e_inv
    return coefs, eig_a


def mult_seq(a_c, a_e, b_c, b_e):
    """
    Multiplies the probabilities of two
    markov chain probability sequences
    to get the overall prob
    of the event. 
    args:
        a_c: Winning sequence coefficients.
        a_e: Winning sequence eigen values.
        b_c: Losing sequence coefficients.
        b_e: Losing sequence eigen values.
    """
    reslt = 0
    n_powers_a = get_n_powers(a_e)
    n_powers_b = get_n_powers(b_e)
    for i in range(len(a_c)):
        for j in range(len(b_c)):
            term = a_c[i]*b_c[j]
            eig_pdt = a_e[i]*b_e[j]
            #n = int(n_powers_a[i] + n_powers_b[j])
            m = int(n_powers_a[i])
            l = int(n_powers_b[j])
            if abs(term) > 1e-5:
                #term = term * (eig_pdt) /(1-eig_pdt)
                #lmb_term = sum_inf_n_powklambda_pown(eig_pdt, n)
                lmb_term = asymmetric_sum(eig_pdt, l, m)
                term = term * lmb_term
            reslt += term
    return reslt


def asymmetric_sum(lmb, l, m):
    reslt = 0.0
    if l+m == 0:
        return lmb/(1-lmb)
    for r in range(m+1):
        inf_term = sum_inf_n_powklambda_pown(lmb,r+l)
        if r+l == 0:
            inf_term -= 1
        reslt += ncr(m,r)*pow(-1,r+m)*inf_term            
    return reslt


def flip_seq(a_c, a_e):
    """
    Flips a sequence from the probabilities
    of being in a state to not benig in the state.
    """
    a_c[a_e<1] = -a_c[a_e<1]
    a_c[a_e==1] = 1-a_c[a_e==1]
    #mult_seq(a_c, a_e, b_c, b_e)
    return a_c


def run_prob_3heads_b4_2heads():
    a = np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]])
    a_c, a_e = get_coefs(a)
    b = np.matrix([[.5,.5,0,0],[.5,0,.5,0],[0.5,0.0,0,.5],[0,0,0,1]])
    b_c, b_e = get_coefs(b,2)
    ## We don't want the first mc to reach it's absorbing state.
    p_n = np.array([sum(a_c*a_e**i) for i in range(2,81)])
    q_n_minus_1 = np.array([sum(b_c*b_e**i) for i in range(1,80)])
    return sum((1-p_n)*q_n_minus_1)/2


def run_prob_3heads_b4_2heads_v2():
    lose_seq = MarkovSequence(get_consecutive_heads_mat(3))
    win_seq = MarkovSequence(get_consecutive_heads_mat(4))
    return get_winner_prob(win_seq, lose_seq)


def run_prob_2running_b4_3consecutive():
    three_total_mat = get_running_total_heads_mat(4)
    two_consecutive_mat = get_consecutive_heads_mat(3)
    start1 = np.array([1,0,0,0])
    start2 = np.array([1,0,0])
    p_n = np.array([np.dot(start1, np.linalg.matrix_power\
                         (three_total_mat,i))[0,3]\
                         for i in range(1,41)])
    q_n_minus_1 = np.array([np.dot(start2, np.linalg.matrix_power\
                            (two_consecutive_mat,i))[0,1]\
                         for i in range(40)])
    return sum((1-p_n)*q_n_minus_1)/2


def run_prob_3running_b4_2running():
    thr_running_mat = get_running_total_heads_mat(4)
    two_running_mat = get_running_total_heads_mat(3)
    start1 = np.array([1,0,0,0])
    start2 = np.array([1,0,0])
    pn = np.array([np.dot(start2, np.linalg.matrix_power\
                         (two_running_mat,i))[0,2]\
                         for i in range(1,31)])
    q_n_minus_1 = np.array([np.dot(start1, np.linalg.matrix_power\
                         (thr_running_mat,i))[0,2]\
                         for i in range(30)])
    return sum((1-pn)*q_n_minus_1)/2


def get_probs(p1=0.5, p2=0.5, n1=3, n2=4, tosses=100):
    """
    Gets the probabilities of 3 consecutive heads before 2,
    2 consecutive before 3 and draw. 
    args:
        p1: P(heads) for the player needing 2 heads.
        p2: P(heads) for the player needing 3 heads.
    """
    start1 = np.zeros(n1)
    start1[0] = 1
    m_3 = get_consecutive_heads_mat(n1,p1)
    start2 = np.zeros(n2)
    start2[0] = 1
    m_4 = get_consecutive_heads_mat(n2,p2)
    # p_n must always be one toss ahead of q_n_minus_1. So, when p_n is at toss 1, q_n_minus_1 
    # should be at 0. When it is at 2, q_n_minus_1 should be at 1 and so on.
    # that is why p_n starts with 1 and goes to 100 while q_n_minus_1 starts with 0 and goes to 99.
    p_n = np.array([np.dot(start1, np.linalg.matrix_power(m_3,n))[0,n1-1]\
                             for n in range(1,tosses+1)])
    q_n_minus_1 = np.array([np.dot(start2, np.linalg.matrix_power(m_4,n))[0,n2-2]\
                             for n in range(tosses)])
    p1wins = sum(q_n_minus_1*(1-p_n))*p2
    p_n = np.array([np.dot(start2, np.linalg.matrix_power(m_4,n))[0,n2-1] for n in range(1,tosses+1)])
    q_n_minus_1 = np.array([np.dot(start1, np.linalg.matrix_power(m_3,n))[0,n1-2] for n in range(tosses)])
    p2wins = sum(q_n_minus_1*(1-p_n))*p1
    return p1wins, p2wins


def plot_probs():
    m_4 = get_consecutive_heads_mat(4)
    m_3 = get_consecutive_heads_mat(3)
    start1 = np.array([1,0,0,0])
    start2 = np.array([1,0,0])
    p_n = np.array([np.dot(start1, np.linalg.matrix_power\
                            (m_4,i))[0,0]\
                         for i in range(40)])
    #plt.plot(np.arange(40), p_n, color='red',label='P(state 0)')
    
    p_n = np.array([np.dot(start1, np.linalg.matrix_power\
                            (m_4,i))[0,1]\
                         for i in range(40)])
    #plt.plot(np.arange(40), p_n, color='yellow',label='P(state 1)')
    p_n = np.array([np.dot(start1, np.linalg.matrix_power\
                            (m_4,i))[0,2]\
                         for i in range(40)])
    #plt.plot(np.arange(40), p_n, color='orange',label='P(state 2)')
    p_n = np.array([np.dot(start1, np.linalg.matrix_power\
                            (m_4,i))[0,3]\
                         for i in range(40)])
    #plt.plot(np.arange(40), p_n, color='green',label='P(state 3)')
    plt.plot(np.arange(40), p_n, color='green',label='P(Seq-1 reaches 3 consecutive heads)')
    plt.xlabel('Number of tosses')
    #plt.ylabel('Probabilities of being in various states')

    q_n = np.array([np.dot(start2, np.linalg.matrix_power\
                            (m_3,i))[0,2]\
                         for i in range(40)])
    plt.plot(np.arange(40), q_n, color='red',label='P(Seq-2 reaches 2 consecutive heads)')
    plt.ylabel('Probs of both seqs being in their target (absorbing) states')
    plt.legend()
    plt.show()


def plot_probs2():
    ps = np.arange(0,1,0.01)
    p_wins = []
    p_losses = []
    p_draws = []
    for p in ps:
        win,loss = get_probs(p2=p)
        p_wins.append(win)
        p_losses.append(loss)
        p_draws.append(1-win-loss)
    plt.plot(ps, p_wins,label='3ConsecHeadsWins')
    plt.plot(ps, p_losses,color='orange',label='2ConsecHeadsWins')
    plt.plot(ps, p_draws,color='silver',label='draw')
    plt.axhline(0.450595,color='grey')
    plt.axvline(0.768482,color='grey')
    plt.axvline(0.5,color='grey')
    plt.xlabel('Probability of heads for sequence needing 3 tosses')
    plt.ylabel('Probabilities of either sequence winning')
    plt.legend()
    plt.show()


def plot_nHb4npl1H():
    ns = np.arange(2,15)
    win_probs = []
    lose_probs = []
    draw_probs = []
    for n in ns:
        lose_seq = MarkovSequence(get_consecutive_heads_mat(n))
        win_seq = MarkovSequence(get_consecutive_heads_mat(n+1))
        ww = win_seq*lose_seq
        ll = lose_seq*win_seq
        win_probs.append(ww)
        lose_probs.append(ll)
        draw_probs.append(1-ll-ww)
    plt.plot(ns, win_probs, label='(n+1) H before n H')
    plt.plot(ns, lose_probs, label='n H before (n+1) H')
    plt.plot(ns, draw_probs,label='draw')
    plt.axhline(0.333,color='grey')
    plt.axhline(0.666,color='grey')
    plt.xlabel('Number of consecutive heads required for smaller sequence')
    plt.ylabel('Probability smaller sequence wins')
    plt.legend()
    plt.show()


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom


def sum_inf_n_powklambda_pown(lmb, k):
    """
    Look in Topics>InfSeries>Sumn^klmb^n
    """
    if k == 0:
        return 1/(1-lmb)
    res = 0.0
    for r in range(k):
        res += ncr(k,r)*sum_inf_n_powklambda_pown(lmb, r)
    return res*lmb/(1-lmb)


def get_generalized_coefficients(trnsn_mat, eigs=None):
    n = trnsn_mat.shape[0]
    if eigs is None:
        eigs = np.linalg.eig(trnsn_mat)[0]
    start_state = np.zeros(n)
    start_state[0] = 1.0 #Assume system starts in most primitive state.
    rhs = [start_state]
    nu_eig_pows = get_n_powers(eigs)
    lhs_col = np.ones(n)
    lhs_col[nu_eig_pows>0]=0
    lhs = [lhs_col]    
    for i in range(n-1):
        start_state = np.array(np.dot(start_state,trnsn_mat))[0]
        rhs.append(start_state)
        lhs_col = eigs**(i+1)
        lhs_col = lhs_col*(i+1)**nu_eig_pows
        lhs.append(lhs_col)
    rhs = np.matrix(rhs)
    lhs = np.matrix(lhs)
    # Pre-multiple [lm1**n, n*lm1**n,...] with this matrix.
    return np.linalg.solve(lhs, rhs)


def get_n_powers(eig):
    n = len(eig)
    enn_x = np.zeros(n)
    x = 1
    for i in range(n):
        if eig[i] == eig[i-1]:
            enn_x[i] = x
            x+=1
        else:
            x=1
    return enn_x


#def tst_matrices():
mm = get_running_total_heads_mat(3, .5)
mm1 = get_running_total_heads_mat(4, .5)
m__ = get_consecutive_heads_mat(3, .5)
m1 = get_consecutive_heads_mat(4, .5)


