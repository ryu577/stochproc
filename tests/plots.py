import numpy as np
from scipy.stats import poisson
from stochproc.hypothesis.hypoth_tst_simulator import run_simulns
from stochproc.count_distributions.compound_poisson import CompoundPoisson


import stochproc.hypothesis.rate_test as xtst
import stochproc.qq_plots.compound_poisson_distribns as cpd
import stochproc.hypothesis.neg_binom_tst as nbt

import matplotlib.pyplot as plt
import seaborn as sns


def plot_alpha_hats_w_alpha_determinst_poisson():
	alphas1,alpha_hats1,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
												lmb=10.0,t1=10,t2=10,l=1)
	alphas2,alpha_hats2,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
												lmb=10.0,t1=10,t2=10,l=2)
	alphas3,alpha_hats3,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
												lmb=10.0,t1=10,t2=10,l=3)
	alphas4,alpha_hats4,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
												lmb=10.0,t1=10,t2=10,l=4)
	alphas5,alpha_hats5,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
												lmb=10.0,t1=10,t2=10,l=5)
	alphas6,alpha_hats6,pois_mas = xtst.UMPPoisson.alpha_on_determinist_compound_closed_form(\
												lmb=10.0,t1=10,t2=10,l=6)

	color_list = sns.color_palette("RdBu_r", 6)
	color_list = color_list.as_hex()
	plt.plot(alpha_hats1,alphas1,label='l=1',color=color_list[0])
	plt.plot(alpha_hats2,alphas2,label='l=2',color=color_list[1])
	plt.plot(alpha_hats3,alphas3,label='l=3',color=color_list[2])
	plt.plot(alpha_hats4,alphas4,label='l=4',color=color_list[3])
	plt.plot(alpha_hats5,alphas5,label='l=5',color=color_list[4])
	plt.plot(alpha_hats6,alphas6,label='l=6',color=color_list[5])
	plt.legend()
	plt.xlabel("Type-1 error rate")
	plt.ylabel("False positive rate")
	plt.show()


def qq_plot():
	cpd.CompBinom.make_qq_plot(0.5,10)


def binom_comp_poisson_alpha_beta():
	dist_rvs_poisson = lambda lmb,t: poisson.rvs(lmb*t)
	alphas1,betas1,alpha_hats1 = run_simulns(fn=dist_rvs_poisson)
	print("Poisson done!")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,2,.4,compound='binom')
	alphas2,betas2,alpha_hats2 = run_simulns(fn=dist_rvs_compound, n_sim=50000)
	print("l=2 done")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,3,.4,compound='binom')
	alphas3,betas3,alpha_hats3 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
	print("l=3 done")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,4,.4,compound='binom')
	alphas4,betas4,alpha_hats4 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
	print("l=4 done")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,5,.4,compound='binom')
	alphas5,betas5,alpha_hats5 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
	print("l=5 done")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,6,.4,compound='binom')
	alphas6,betas6,alpha_hats6 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
	print("l=5 done")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,10,.4,compound='binom')
	alphas10,betas10,alpha_hats10 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
	print("l=10 done")
	dist_rvs_compound = lambda lmb,t: CompoundPoisson.rvs_s(lmb*t,20,.4,compound='binom')
	alphas20,betas20,alpha_hats20 = run_simulns(fn=dist_rvs_compound, n_sim=5000)
	print("all done")

	color_list = sns.color_palette("RdBu_r", 20)
	color_list = color_list.as_hex()

	plt.plot(alphas1,betas1,label='UMP poisson on poisson',color='black')
	plt.plot(alphas2,betas2,label='UMP poisson on compound poisson l=2',color=color_list[1])
	plt.plot(alphas3,betas3,label='UMP poisson on compound poisson l=3',color=color_list[2])
	plt.plot(alphas4,betas4,label='UMP poisson on compound poisson l=4',color=color_list[3])
	plt.plot(alphas5,betas5,label='UMP poisson on compound poisson l=5',color=color_list[4])
	plt.plot(alphas6,betas6,label='UMP poisson on compound poisson l=6',color=color_list[5])
	plt.plot(alphas10,betas10,label='UMP poisson on compound poisson l=10',color=color_list[9])
	plt.plot(alphas20,betas20,label='UMP poisson on compound poisson l=20',color=color_list[19])
	plt.xlabel('Alpha')
	plt.ylabel('Beta')
	plt.legend()
	plt.show()

	plt.plot(alpha_hats1,alphas1,label='UMP poisson on poisson')
	plt.plot(alpha_hats2,alphas2,label='UMP poisson on compound poisson l=2',color=color_list[1])
	plt.plot(alpha_hats3,alphas3,label='UMP poisson on compound poisson l=3',color=color_list[2])
	plt.plot(alpha_hats4,alphas4,label='UMP poisson on compound poisson l=4',color=color_list[3])
	plt.plot(alpha_hats5,alphas5,label='UMP poisson on compound poisson l=5',color=color_list[4])
	plt.plot(alpha_hats6,alphas6,label='UMP poisson on compound poisson l=6',color=color_list[5])
	plt.plot(alpha_hats10,alphas10,label='UMP poisson on compound poisson l=10',color=color_list[9])
	plt.plot(alpha_hats20,alphas20,label='UMP poisson on compound poisson l=20',color=color_list[19])
	plt.xlabel('Type-1 error')
	plt.ylabel('False positive rate')
	plt.legend()
	plt.show()


def binom_tst_on_nbd_beta():
	ms = np.array([10,20,30,40,50,60,70,80,90,100,120,150,200,250,320,500,700,1000])
	betas_m = []
	##Changing m
	for m in ms:
		beta_m_10 = xtst.UMPPoisson.beta_on_negbinom_closed_form(t1=30,t2=30,\
		                theta_base=5,m=m,deltheta=1,alpha=0.05,cut_dat=1e4)[0]
		betas_m.append(beta_m_10)
	plt.xlabel("The m parameter of both groups")
	plt.ylabel("False negative rate for type-1 error=0.05")
	plt.plot(ms,betas_m)


	ts = np.array([10,20,30,40,50,60,70,80,90,100,120,150,200,250,320,500,700,1000,1500,2000,4000])
	betas_t = []
	##Changing m
	for t in ts:
		beta_t_10 = xtst.UMPPoisson.beta_on_negbinom_closed_form(t1=t,t2=t,\
		                theta_base=5,m=10,deltheta=1,alpha=0.05,cut_dat=1e4)[0]
		betas_t.append(beta_t_10)

	plt.xlabel("Time period of observation in both groups")
	plt.ylabel("False negative rate for type-1 error=0.05")
	plt.plot(ts,betas_t)
	plt.show()




