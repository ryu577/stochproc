import numpy as np


def series(lmbs, mus):
    fail_rate = sum(lmbs)
    tmp_arr = mus/(mus+lmbs)
    factor = np.prod(tmp_arr)/(1-np.prod(tmp_arr))
    repair_rate = fail_rate*factor
    reliability = repair_rate/(repair_rate+fail_rate)
    return fail_rate, repair_rate, reliability


def parallel(lmbs, mus):
    repair_rate = sum(mus)
    tmp_arr = lmbs/(mus+lmbs)
    factor = np.prod(tmp_arr)/(1-np.prod(tmp_arr))
    fail_rate = factor*repair_rate
    reliability = repair_rate/(repair_rate+fail_rate)
    return fail_rate, repair_rate, reliability



