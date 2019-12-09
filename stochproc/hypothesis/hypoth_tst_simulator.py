import numpy as np
from scipy.stats import binom_test, poisson, binom
from scipy.special import gamma
from stochproc.count_distributions.compound_poisson import CompoundPoisson
from stochproc.count_distributions.interarrival_weibull import InterarrivalWeibull
from stochproc.hypothesis.rate_test import rateratio_test
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


def alpha_beta_curve(rvs_fn, n_sim=10000, lmb=20, t1=10, t2=3, scale=1.0):
    ## First the null hypothesis..
    alpha_hats = np.concatenate((np.arange(0.000000000001,0.0099,0.000001), 
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


def run_simulns(fn, n_sim=50000, lmb=20.0, t1=10.0, t2=3.0, scale=1.0):
    time1=datetime.now()
    alphas1,betas1,alpha_hats1 = alpha_beta_curve(fn, 
                                    n_sim=n_sim, 
                                    lmb=lmb,
                                    t1=t1, t2=t2, 
                                    scale=scale)
    time2=datetime.now()
    time_del = (time2-time1).seconds
    print("Time taken in seconds: " + str(time_del))
    return alphas1, betas1, alpha_hats1



# mpl.rcParams.update({'text.color' : "white",
#                         'axes.labelcolor' : "white",
#                         'xtick.color' : "white",
#                         'ytick.color' : "white",
#                         "axes.edgecolor" : "white"})

# fig, ax = plt.subplots(facecolor='black')
# #ax.set_axis_bgcolor("black")
# ax.set_facecolor("black")
fig, ax = plt.subplots()

def plot_tests_on_distributions():
    
    alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_poisson)
    alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_compound, n_sim=50000)
    alphas3,betas3,alpha_hats3 = run_simulns(fn=dist_rvs_interarrivalw, n_sim=5000)
    alphas4,betas4,alpha_hats4 = run_simulns(fn=dist_rvs_interarrivalw, n_sim=5000, scale=25.0)
    alphas5,betas5,alpha_hats5 = run_simulns(fn=dist_rvs_interarrivalw, n_sim=5000, scale=1/10.0)

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
    alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_poisson)
    alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_compound, n_sim=50000)
    plt.plot(alpha_hats1,alphas1,label='UMP poisson on poisson')
    plt.plot(alpha_hats2,alphas2,label='UMP poisson on compound poisson')
    plt.xlabel('Alpha you set')
    plt.ylabel('Alpha you get')
    plt.legend(facecolor="black", edgecolor="black")
    fig.savefig("C:\\Users\\ropandey\\OneDrive\\MSFTProj\\HypothTestAIR\\alpha_hat_w_alpha.png", \
        facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


def plot_alpha_beta():
    alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_poisson)
    alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_compound, n_sim=50000)
    plt.plot(alpha_hats1,alphas1,label='UMP poisson on poisson')
    plt.plot(alpha_hats2,alphas2,label='UMP poisson on compound poisson')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.legend(facecolor="black", edgecolor="black")
    fig.savefig("C:\\Users\\ropandey\\OneDrive\\MSFTProj\\HypothTestAIR\\alpha_hat_w_alpha.png", \
        facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


def fit_alpha_profile(alpha_hats2,alphas2):
    zz=np.polyfit(alpha_hats2,alphas2,8)
    alpha_mat = np.array([alpha_hats2**8,alpha_hats2**7,alpha_hats2**6,alpha_hats2**5,alpha_hats2**4,\
            alpha_hats2**3,alpha_hats2**2,alpha_hats2**1,alpha_hats2**0]).T
    alpha_pred = np.dot(alpha_mat,zz)
    plt.plot(alpha_hats2,alphas2,label='actual')
    plt.plot(alpha_hats2,alpha_pred,label='polynomial')


## Does the mapping between alpha-hat and alpha change 
## if we have different observation windows in each group?
def alpha_plots():
    alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_compound, n_sim=5000, t1=10.0,t2=3.0)
    alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_compound, n_sim=5000, t1=3.0,t2=10.0)
    alphas3,betas3,alpha_hats3 = run_simulns(fn=dist_rvs_compound, n_sim=5000, t1=10.0,t2=10.0)
    alphas4,betas4,alpha_hats4 = run_simulns(fn=dist_rvs_compound, n_sim=5000, t1=3.0,t2=3.0)
    alphas5,betas5,alpha_hats5 = run_simulns(fn=dist_rvs_compound, n_sim=5000, lmb=5.0, t1=10.0,t2=10.0)

    plt.plot(alpha_hats1,alphas1,label="t1=10; t2=3; lmb=20")
    plt.plot(alpha_hats2,alphas2,label="t1=3; t2=10; lmb=20")
    plt.plot(alpha_hats3,alphas3,label="t1=10; t2=10; lmb=20")
    plt.plot(alpha_hats4,alphas4,label="t1=3; t2=3; lmb=20")
    plt.plot(alpha_hats5,alphas5,label="t1=10; t2=10; lmb=5")

    plt.xlabel('Alpha you set')
    plt.ylabel('Alpha you get')
    plt.legend(facecolor="black", edgecolor="black")
    fig.savefig("C:\\Users\\rohit\OneDrive\\MSFTProj\\HypothTestAIR\\alpha_mapping_no_change.png", \
                    facecolor=fig.get_facecolor(), transparent=True)
    plt.close()


def plot_hypotheses_on_distributions():
    alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_poisson)
    alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_poisson,scale=2.0)
    alphas3,betas3,alpha_hats3 = run_simulns(fn=dist_rvs_poisson,scale=0.5)
    alphas4,betas4,alpha_hats4 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
    alphas5,betas5,alpha_hats5 = run_simulns(fn=dist_rvs_compound, n_sim=5000,scale=0.3)
    alphas6,betas6,alpha_hats6 = run_simulns(fn=dist_rvs_compound, n_sim=5000,scale=22.4)

    plt.plot(alpha_hats1,alphas1,label="sc=1.0 on Poisson")
    #plt.plot(alpha_hats2,alphas2,label="sc=2.0")
    #plt.plot(alpha_hats3,alphas3,label="sc=0.5")
    plt.plot(alpha_hats4,alphas4,label="sc=1.0 on Compound Poisson")
    #plt.plot(alpha_hats5,alphas5,label="sc=0.3 on CPP")
    plt.plot(alpha_hats6,alphas6,label="sc=22.4 on Compound Poisson")
    plt.legend()
    plt.show()
    plt.plot(alphas1,betas1,label='UMP poisson on poisson')
    plt.plot(alphas2,betas2,label='UMP poisson; sc:2 on poisson')
    plt.plot(alphas3,betas3,label='UMP poisson; sc:0.5 on poisson')
    plt.plot(alphas4,betas4,label='UMP poisson on compound poisson')
    plt.plot(alphas5,betas5,label='UMP poisson sc:0.3 on compund poisson')
    plt.plot(alphas6,betas6,label='UMP poisson sc:22.4 on compund poisson')
    plt.show()
    ## Theoretical alpha-beta profile.
    alpha_hats = np.arange(0,1,0.0001)
    a = binom.isf(alpha_hats,50,0.5)
    betas = binom.cdf(a,50,0.65)
    plt.plot(betas,alpha_hats)
    plt.show()

