from stochproc.quantile.estimate import *
from stochproc.quantile.expon_based_estimators import expon_frac, prcntl
from stochproc.quantile.some_distributions import *
import numpy as np


class PrcntlEstPerfMeasurer():
    def __init__(self, n=15,
                 rvs_fn=rvs_fn2,
                 ppf_fn=ppf_fn2,
                 qs=np.arange(0.3, 0.7, 0.03),
                 prcntl_estimator=prcntl,
                 prll_wrlds=30000,
                 verbose=True):
        self.n = n
        self.rvs_fn = rvs_fn
        self.ppf_fn = ppf_fn
        self.qs = qs
        self.prcntl_estimator = prcntl_estimator
        self.prll_wrlds = prll_wrlds
        self.verbose = verbose

    def simulate(self):
        self.u_errs = []
        self.u_stds = []
        self.u_coff_vars = []
        self.u_medians = []
        self.u_mses = []

        for q in self.qs:
            errs = []
            ests = []
            for _ in range(self.prll_wrlds):
                x = self.rvs_fn(self.n)
                real_val = self.ppf_fn(q)
                est_val = self.prcntl_estimator(x, q)
                err = (real_val-est_val)
                errs.append(err)
                ests.append(est_val)
            self.u_errs.append(np.mean(errs))
            self.u_stds.append(np.std(ests))
            self.u_medians.append(np.median(errs))
            self.u_coff_vars.append(np.std(ests)/np.mean(ests))
            self.u_mses.append(np.sqrt(np.var(ests)+np.mean(errs)**2))
            if self.verbose:
                print("Finished processing percentile: " + str(q))
