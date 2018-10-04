import numpy as np
import abc


class MarkovSequence:
    def __init__(self, trnsn_matrix):
        self.trnsn_matrix = trnsn_matrix
        self.seq_len = trnsn_matrix.shape[0]
        self.get_coef_matrix()

    def __mul__(self, other):
        return get_winner_prob(self, other)

    def __rmul__(other, self):
        return get_winner_prob(other, self)

    def get_coef_matrix(self):
        a = self.trnsn_matrix
        [eig_a, eig_vec_a] = np.linalg.eig(a)
        eig_vec_a_inv = np.linalg.inv(eig_vec_a)
        first_row_e = np.array(eig_vec_a[0])[0]
        self.coef_matrix = np.dot(np.diag(first_row_e),\
                                   eig_vec_a_inv)
        self.eigs = eig_a
        self.ultimate_coefs = np.array(self.coef_matrix.T[self.seq_len-1])[0]
        self.penultimate_coefs = np.array(self.coef_matrix.T[self.seq_len-2])[0]



def get_winner_prob(win_seq, lose_seq):
    a_c, a_e = win_seq.penultimate_coefs, win_seq.eigs
    a_c = a_c/a_e
    #a_c = flip_seq(a_c, a_e)
    b_c, b_e = lose_seq.ultimate_coefs, lose_seq.eigs
    b_c = flip_seq(b_c, b_e)
    #b_c = b_c/b_e
    ans = mult_seq(a_c, a_e, b_c, b_e)
    return ans/2


def get_consecutive_heads_mat(numstates=3):
    trnsn_mat = np.matrix(np.zeros((numstates,numstates)))
    for i in range(trnsn_mat.shape[0]-1):
        trnsn_mat[i,0] = 0.5
        trnsn_mat[i,i+1] = 0.5
    trnsn_mat[numstates-1,numstates-1] = 1.0
    return trnsn_mat


def get_running_total_heads_mat(numstates=4):
    trnsn_mat = np.eye(numstates)*0.5
    for i in range(numstates-1):
        trnsn_mat[i,i+1] = 0.5
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
    """
    reslt = 0
    for i in range(len(a_c)):
        for j in range(len(b_c)):
            term = (a_c[i]*b_c[j])
            if abs(term) > 1e-5:
                term = term * (a_e[i]*b_e[j]) /(1-a_e[i]*b_e[j])            
            reslt += term
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


