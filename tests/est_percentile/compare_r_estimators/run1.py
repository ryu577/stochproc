from stochproc.quantile.estimate import *
from stochproc.quantile.expon_based_estimators import expon_frac, prcntl
from stochproc.quantile.some_distributions import *
from stochproc.quantile.perf_measurer import PrcntlEstPerfMeasurer
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os


## Set these parameters manually before running the code.
data_save_dir = "C:\\Users\\Administrator\\Documents\\github\\stochproc\\tests\\est_percentile\\compare_r_estimators\\sim_data\\"
plots_save_dir = "C:\\Users\\Administrator\\Documents\\github\\stochproc\\tests\\est_percentile\\compare_r_estimators\\plots\\"
rvs_fn = rvs_fn2
ppf_fn = ppf_fn2
distr_name = "LogNormal"
# The percentiles to compare performance for.
qs = np.arange(0.01, 1.0, 0.03)


# Enumerate the estimators.
def main1():
    prcntl_estimators = [prcntl, est_1, est_7,
                         est_2, est_3, est_4,
                         est_5, est_6,
                         est_8, est_9]

    names = ["expon_bias", "r_strat1",
             "r_strat7",
             "r_strat2", "r_strat3",
             "r_strat4", "r_strat5",
             "r_strat6",
             "r_strat8", "r_strat9"]

    prf_results = []

    #fig1, (ax1, ax3) = plt.subplots(2, 1)
    #fig2, (ax2, ax4) = plt.subplots(2, 1)
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)
    fig4, ax4 = plt.subplots(1, 1)

    for ix in range(len(prcntl_estimators)):
        prcntl_est = prcntl_estimators[ix]
        name = names[ix]
        prf1 = PrcntlEstPerfMeasurer(n=15,
                                     rvs_fn=rvs_fn,
                                     ppf_fn=ppf_fn,
                                     qs=qs,
                                     prcntl_estimator=prcntl_est,
                                     prll_wrlds=30000)
        prf1.simulate()
        prf_results.append(prf1)
        ax1.plot(qs, prf1.u_errs, label="Bias for " + name)
        ax2.plot(qs, prf1.u_stds, label="Standard deviation for " + name)
        ax3.plot(qs, prf1.u_medians, label="DelMedian for " + name)
        ax4.plot(qs, prf1.u_mses, label="MSE for " + name)

        np.savetxt(data_save_dir + "\\qs.csv", qs, delimiter=",")
        base_path = data_save_dir + distr_name + "\\" + name
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        np.savetxt(data_save_dir + distr_name + "\\" +
                   name + "\\u_errs.csv", prf1.u_errs, delimiter=",")
        np.savetxt(data_save_dir + distr_name + "\\" +
                   name + "\\u_stds.csv", prf1.u_stds, delimiter=",")
        np.savetxt(data_save_dir + distr_name + "\\" +
                   name + "\\u_medians.csv", prf1.u_medians, delimiter=",")
        np.savetxt(data_save_dir + distr_name + "\\" +
                   name + "\\u_mses.csv", prf1.u_mses, delimiter=",")

        print("###############")
        print("Processed " + name)
        print("###############")

    make_lines(ax1, ax2, ax3, ax4)
    base_path = plots_save_dir + distr_name + "\\" + name
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    fig1.savefig(plots_save_dir + distr_name + "\\" + name + "\\biases.png")
    fig2.savefig(plots_save_dir + distr_name + "\\" + name + "\\st_devs.png")
    fig3.savefig(plots_save_dir + distr_name + "\\" +
                 name + "\\del_medians.png")
    fig4.savefig(plots_save_dir + distr_name + "\\" + name + "\\mses.png")
    plt.xlabel("Percentile (q)")
    plt.show()
    return prf_results


def make_plots_from_disk():
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)
    fig4, ax4 = plt.subplots(1, 1)
    names = ["expon_bias", "r_strat1",
             "r_strat7",
             "r_strat2", "r_strat3",
             "r_strat4", "r_strat5",
             "r_strat6",
             "r_strat8", "r_strat9"]
    qs = genfromtxt(data_save_dir + "\\qs.csv", delimiter=",")
    for name in names:
        u_errs = genfromtxt(data_save_dir + distr_name + "\\" +
                            name + "\\u_errs.csv", delimiter=',')
        u_stds = genfromtxt(data_save_dir + distr_name + "\\" +
                            name + "\\u_stds.csv", delimiter=',')
        u_medians = genfromtxt(data_save_dir + distr_name + "\\" +
                               name + "\\u_medians.csv", delimiter=',')
        u_mses = genfromtxt(data_save_dir + distr_name + "\\" +
                            name + "\\u_mses.csv", delimiter=',')
        ax1.plot(qs, u_errs, label="Bias for " + name)
        ax2.plot(qs, u_stds, label="Standard deviation for " + name)
        ax3.plot(qs, u_medians, label="DelMedian for " + name)
        ax4.plot(qs, u_mses, label="MSE for " + name)
    make_lines(ax1, ax2, ax3, ax4)
    plt.show()


def make_lines(ax1, ax2, ax3, ax4):
    ax1.axhline(0, color="black")
    ax1.axvline(0.5, color="black")
    ax2.axhline(0, color="black")
    ax2.axvline(0.5, color="black")
    ax3.axhline(0, color="black")
    ax3.axvline(0.5, color="black")
    ax4.axhline(0, color="black")
    ax4.axvline(0.5, color="black")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
