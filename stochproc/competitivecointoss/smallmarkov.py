import numpy as np
import abc


class MarkovSequence:
    def __init__(self, trnsn_matrix):
        self.trnsn_matrix = trnsn_matrix
        self.get_coef_matrix()

    def get_coef_matrix(self):
        a = self.trnsn_matrix
        [eig_a, eig_vec_a] = np.linalg.eig(a)
        eig_vec_a_inv = np.linalg.inv(eig_vec_a)
        first_row_e = np.array(eig_vec_a[0])[0]
        self.coef_matrix = np.dot(fnp.diag(irst_row_e), eig_vec_a_inv)


def get_consecutive_heads_markov_mat(numstates=3):
    trnsn_mat = np.matrix(np.zeros((3,3)))
    for i in range(trnsn_mat.shape[0]-1):
        trnsn_mat[i,0] = 0.5
        trnsn_mat[i,i+1] = 0.5
    trnsn_mat[numstates-1,numstates-1] = 1.0


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
            term = a_c[i]*b_c[j]
            inf_fac = 1
            if a_e[i] < 1:
                inf_fac *= a_e[i]
            if b_e[j] < 1:
                inf_fac *= b_e[j]
            if inf_fac < 1:
                term *= 1/(1-inf_fac)
            reslt += term
    return reslt


def run_prob_3heads_b4_2heads():
    a = np.matrix([[.5,.5,0],[.5,0,.5],[0,0,1]])
    a_c, a_e = get_coefs(a)
    b = np.matrix([[.5,.5,0,0],[.5,0,.5,0],[0.5,0.0,0,.5],[0,0,0,1]])
    b_c, b_e = get_coefs(b,2)
    ## We don't want the first mc to reach it's absorbing state.
    p_n = np.array([sum(a_c*a_e**i) for i in range(2,81)])
    q_n_minus_1 = np.array([sum(b_c*b_e**i) for i in range(1,80)])
    return sum((1-p_n)*q_n_minus_1)/2


def flip_seq(a_c, a_e):
    """
    Flips a sequence from the probabilities
    of being in a state to not benig in the state.
    """
    a_c[a_e<1] = -a_c[a_e<1]
    a_c[a_e==1] = 1-a_c[a_e==1]
    #mult_seq(a_c, a_e, b_c, b_e)
    return a_c



