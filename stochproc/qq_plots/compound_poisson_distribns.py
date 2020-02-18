import numpy as np
from scipy.stats import poisson, binom
from stochproc.count_distributions.compound_poisson import CompoundPoisson
import matplotlib.pyplot as plt
import pyray.plotting.matplot_utils as matplot_utils
import seaborn as sns

class CompBinom():
    @staticmethod
    def qq_comp_binom_poiss_comp_determn_poiss(l,p,poisson_mu=4.0,n_sim=1000,\
                                            alpha_hats = np.arange(0.0001,1.0,1e-3)):
        alphas = np.zeros(len(alpha_hats))
        isfs = int(l*p)*poisson.isf(alpha_hats,poisson_mu)
        for _ in range(n_sim):
            comp_binom_rv = CompoundPoisson.rvs_s(poisson_mu,l,p,compound='binom')
            alphas += (comp_binom_rv > isfs)/n_sim
        return alphas, alpha_hats

    ##TODO: Move this to pyray. Also, fix it, it doesn't work!
    @staticmethod
    def make_matplotlib_dark():
        mpl.rcParams.update({'text.color' : "white",
                                 'axes.labelcolor' : "white",
                                 'xtick.color' : "white",
                                 'ytick.color' : "white",
                                 "axes.edgecolor" : "white"})

        fig, ax = plt.subplots(facecolor='black')
        ax.set_axis_bgcolor("black")
        ax.set_facecolor("black")
        fig, ax = plt.subplots()

    @staticmethod
    def make_qq_plot(p,poisson_mu,black_plot=False,use_heatmap=True,l_vals=None):
        if use_heatmap and l_vals is None:
            l_vals = np.array([3,5,7,11,13,17,19,23,31,37,41,43,47,53,59,61,67,71,73,79,83,101,113,137])
        elif l_vals is None:
            l_vals = np.array([3,5,7,11,13,17,19,23,31])
        # See: https://seaborn.pydata.org/tutorial/color_palettes.html
        color_list = sns.color_palette("RdBu_r", len(l_vals))
        color_list = color_list.as_hex()
        for l in range(len(l_vals)):
            alphas, alpha_hats = CompBinom.qq_comp_binom_poiss_comp_determn_poiss(\
                                            l_vals[l],p,poisson_mu)
            if not use_heatmap:
                plt.plot(alpha_hats, alphas, label='l= '+str(l_vals[l]))#,color=color_list[l])
            else:
                plt.plot(alpha_hats, alphas, label='l= '+str(l_vals[l]),color=color_list[l])
        p1 = [0,0]
        p2 = [1,1]
        matplot_utils.newline(p1,p2)
        plt.legend()
        #plt.title("QQ plot between compound binomial and deterministically \
        #                compounded process")
        plt.xlabel("Deterministically compounded Poisson quantile")
        plt.ylabel("Binomially compounded Poisson quantile")
        plt.show()

    @staticmethod
    def plot_l_deltas(p=0.5,poisson_mu=100.0,plot=True):
        #l_vals = np.array([3,5,7,11,13,17,19,23,31,37,41,43,47,53,59,61,67,71,73,79,83,101,113,137])
        l_vals = np.arange(3,100,1)
        alpha_hats = np.arange(0.0001,1.0,1e-3)
        deviatns = np.zeros(len(l_vals))
        for i in range(len(l_vals)):
            alphas, alpha_hats = CompBinom.qq_comp_binom_poiss_comp_determn_poiss(\
                                            l_vals[i],p,poisson_mu,alpha_hats=alpha_hats)
            deviatns[i] = sum((alphas-alpha_hats)**2)
        if plot:
            plt.plot(l_vals,deviatns)
            plt.title("For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))
            plt.xlabel("Binomial num tosses: l")
            plt.ylabel("SSE between alpha and alpha_hat")
            plt.show()
        return l_vals, deviatns

    @staticmethod
    def plot_l_deltas_for_configurations():
        p=0.1; poisson_mu=17
        ls, devtns = CompBinom.plot_l_deltas(p=p,poisson_mu=poisson_mu,plot=False)
        plt.plot(ls, devtns,label="For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))
        p=0.14285714; #poisson_mu=60
        ls, devtns = CompBinom.plot_l_deltas(p=p,poisson_mu=poisson_mu,plot=False)
        plt.plot(ls, devtns,label="For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))
        p=0.2; #poisson_mu=60
        ls, devtns = CompBinom.plot_l_deltas(p=p,poisson_mu=poisson_mu,plot=False)
        plt.plot(ls, devtns,label="For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))
        p=0.25; #poisson_mu=60
        ls, devtns = CompBinom.plot_l_deltas(p=p,poisson_mu=poisson_mu,plot=False)
        plt.plot(ls, devtns,label="For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))
        p=0.33333; #poisson_mu=60
        ls, devtns = CompBinom.plot_l_deltas(p=p,poisson_mu=poisson_mu,plot=False)
        plt.plot(ls, devtns,label="For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))        
        p=0.5; #poisson_mu=60
        ls, devtns = CompBinom.plot_l_deltas(p=p,poisson_mu=poisson_mu,plot=False)
        plt.plot(ls, devtns,label="For Poisson mu:" + str(poisson_mu) + " and binomial p= " + str(p))
        plt.legend()
        plt.show()


