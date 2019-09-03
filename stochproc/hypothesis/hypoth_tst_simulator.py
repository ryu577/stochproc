import numpy as np
from scipy.stats import binom_test, poisson
from scipy.special import gamma
from stochproc.count_distributions.compound_poisson import CompoundPoisson
from stochproc.count_distributions.interarrival_weibull import InterarrivalWeibull
import matplotlib.pyplot as plt
from datetime import datetime

def rateratio_test(n1,t1,n2,t2,scale=1.0):
    n2, n1 = n2/scale, n1/scale
    p_val = binom_test(n2,n1+n2,t2/(t1+t2),alternative='greater')
    return p_val


def alpha_beta_curve(rvs_fn, n_sim=10000, lmb=20, t1=10, t2=3, scale=1.0):
    ## First the null hypothesis..
    alpha_hats = np.concatenate((np.arange(0.000001,0.0099,0.0001), 
                                np.arange(0.01,1.00,0.01), 
                                np.arange(0.991,1.00,0.001)),axis=0)
    alphas = np.zeros(len(alpha_hats))

    ## First generate from null and find alpha_hat and alpha.
    for _ in range(n_sim):
        m1 = rvs_fn(lmb,t1)
        m2 = rvs_fn(lmb,t2)
        p_val = rateratio_test(m1,t1,m2,t2,scale)
        alphas += (p_val < alpha_hats)/n_sim

    ## Now the alternate hypothesis
    dellmb = 10.0
    betas = np.zeros(len(alpha_hats))
    for _ in range(n_sim):
        m1 = rvs_fn(lmb,t1)
        m2 = rvs_fn((lmb+dellmb),t2)
        p_val = rateratio_test(m1,t1,m2,t2,scale)
        betas += 1/n_sim - (p_val < alpha_hats)/n_sim
    return alphas, betas, alpha_hats


n=32; p=0.7
dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,n,p)
dist_rvs_poisson = lambda lmb,t: poisson.rvs(lmb*t)

k=0.5; lmb_target = 20
def dist_rvs_interarrivalw(lmb_target=20,t=20):
    ## This is approximate. 
    w_lmb = 1/lmb_target/gamma(1+1/k)
    iw = InterarrivalWeibull(k,w_lmb,t)
    return iw.rvs1()

def run_simulns(fn, n_sim=50000, scale=1.0):
    t1=datetime.now()
    alphas1,betas1,alpha_hats1 = alpha_beta_curve(fn, n_sim=n_sim, 
                                    scale=scale)
    t2=datetime.now()
    time_del = (t2-t1).seconds
    print("Time taken in seconds: " + str(time_del))
    return alphas1,betas1,alpha_hats1


alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_poisson)
alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
alphas3,betas3,alpha_hats3 = run_simulns(fn=dist_rvs_interarrivalw, n_sim=5000)
alphas4,betas4,alpha_hats4 = run_simulns(fn=dist_rvs_interarrivalw, n_sim=5000, scale=25.0)
alphas5,betas5,alpha_hats5 = run_simulns(fn=dist_rvs_interarrivalw, n_sim=5000, scale=1/10.0)

import matplotlib as mpl

mpl.rcParams.update({'text.color' : "white",
                        'axes.labelcolor' : "white",
                        'xtick.color' : "white",
                        'ytick.color' : "white",
                        "axes.edgecolor" : "white"})

fig, ax = plt.subplots(facecolor='black')
ax.set_axis_bgcolor("black")


def plot_all_combinations():
    plt.plot(alphas1,betas1,label='UMP poisson on poisson')
    plt.plot(alphas2,betas2,label='UMP poisson on compound poisson')
    plt.plot(alphas3,betas3,label='UMP poisson on interarrival weibull')
    plt.plot(alphas4,betas4,label='UMP poisson sc:25.0 on interarrival weibull')
    plt.plot(alphas5,betas5,label='UMP poisson sc:0.1 on interarrival weibull')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.legend(facecolor="black", edgecolor="black")
    fig.savefig("C:\\Users\\rohit\OneDrive\\MSFTProj\\HypothTestAIR\\all_combinations.png", \
        facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


def plot_alpha_with_hat():
    plt.plot(alpha_hats1,alphas1,label='UMP poisson on poisson')
    plt.plot(alpha_hats2,alphas2,label='UMP poisson on compound poisson')
    plt.xlabel('Alpha you set')
    plt.ylabel('Alpha you get')
    plt.legend(facecolor="black", edgecolor="black")
    fig.savefig("C:\\Users\\rohit\OneDrive\\MSFTProj\\HypothTestAIR\\alpha_hat_w_alpha.png", \
        facecolor=fig.get_facecolor(), transparent=True)
    plt.close()

def plot_alpha_beta():
    plt.plot(alpha_hats1,alphas1,label='UMP poisson on poisson')
    plt.plot(alpha_hats2,alphas2,label='UMP poisson on compound poisson')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.legend(facecolor="black", edgecolor="black")
    fig.savefig("C:\\Users\\rohit\OneDrive\\MSFTProj\\HypothTestAIR\\alpha_hat_w_alpha.png", \
        facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


