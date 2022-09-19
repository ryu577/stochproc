# import libraries 
import time
from sim_life import cmp_ests
from sim_birth_death import cmp_ests as cmp_ests_bd
import numpy as np
import pandas as pd
import os
import platform

# common values across all test cases 
WINDOW_SIZES = [(500, 550)]
SMALL_VALS = [1, 3, 5]
LARGE_VALS = [5, 10, 15, 25, 50, 75, 100, 150, 250, 350]
VMS = [1, 10, 100, 500]

# test cases for processes 1 and 2 (vms, s, e, lmb, mu)
TEST_CASES_PROC1, TEST_CASES_PROC2 = [], []
for vms in VMS: 
    for s, e in WINDOW_SIZES:
        for lmb in LARGE_VALS: 
            if vms == 1: 
                TEST_CASES_PROC1.append((1, s, e, lmb, 0))
            else: 
                TEST_CASES_PROC2.append((vms, s, e, lmb, 0))
            
# test cases for process 3 (vms, s, e, lmb, mu)
TEST_CASES_PROC3 = []
for vms in VMS:
    for mu in SMALL_VALS: 
        for s, e in WINDOW_SIZES:
            for lmb in LARGE_VALS: 
                TEST_CASES_PROC3.append((vms, s, e, lmb, mu))

# test cases for process 4 (vms, s, e, lmb, mu)
TEST_CASES_PROC41, TEST_CASES_PROC42 = [], []
for s, e in WINDOW_SIZES: 
    for v1 in SMALL_VALS: 
        for v2 in LARGE_VALS: 
            TEST_CASES_PROC41.append((s, e, v1, v2))
            TEST_CASES_PROC42.append((s, e, v2, v1))
TEST_CASES_PROC4 = TEST_CASES_PROC41 + TEST_CASES_PROC42

# Excel sheet column names for TTX and rate simulations 
METRICS = ['Bias', 'Variance', 'MSE']
TTXS = ['E_1', 'E_2', 'E_5']
TTX_COLUMNS = []
RATES = []
RATES_COLUMNS = []
for ttx in TTXS: 
    RATES.append(f'1/{ttx}')
    for metric in METRICS: 
        TTX_COLUMNS.append(f'{ttx} {metric}')
        RATES_COLUMNS.append(f'1/{ttx} {metric}')

def _run_sims(test_cases, save_file_name, ttx=True):
    '''
    Runs simulations with given settings/parameters. Compiles results into 
    pd.DataFrame and saves to Excel

    Parameters: 
    test_cases - list of test cases 
    cols - list of the names of the estimators 
    res_cols - list of the names of the estimators and the metrics (bias, variance, mse)
    bias_adjustment - what to subtract from expected value estimations to get
    estimated bias ('lmb', 'mu')
    ttx - if result gives ttx, set to True, else rate, set to False
    '''
    # run simulations 
    sim_res_info = [] 
    if len(test_cases[0]) == 5: 
        for vms, s, e, lmb, mu in test_cases: 
            res = cmp_ests_bd(s, e, lmb, mu, vms, ttx)
            if ttx:
                # res = 1 / res # not needed, done in cmp_ests_bd
                res[:, 0] -= lmb
            else:
                res[:, 0] -= 1 / lmb
            res = list(res.flatten())
            sim_res_info.append(res)
    else:
        for s, e, lmb, mu in test_cases: 
            res = cmp_ests(s, e, lmb, mu, ttx)
            if not ttx:
                # res = 1 / res # not needed, done in cmp_ests
                res[:, 0] -= 1 / mu
            else: 
                res[:, 0] -= mu
            res = list(res.flatten())
            sim_res_info.append(res)

    # create DataFrame 
    if ttx: 
        cols = TTXS
        res_cols = TTX_COLUMNS
    else: 
        cols = RATES
        res_cols = RATES_COLUMNS
    str_test_cases = list(map(lambda t: str(t), test_cases))
    sim_res_df = pd.DataFrame(data=sim_res_info, index=str_test_cases,\
        columns=res_cols)
    if len(test_cases[0]) == 5:
        sim_res_df.index.rename('Test case (vms, s, e, lmb, mu)',\
            inplace=True)
    else: 
        sim_res_df.index.rename('Test case (s, e, lmb, mu)',\
            inplace=True)

    # write to excel sheet
    dir = os.path.dirname(os.path.abspath(__file__))
    plat = platform.platform()
    save_path = dir + '/' + save_file_name
    if 'windows' in plat.lower(): 
        save_path = dir + '\\' + save_file_name
    try: 
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            # write results to sheet
            sim_res_df.to_excel(writer, sheet_name=f'Estimations')
            workbook = writer.book
            worksheet = writer.sheets['Estimations']
            opt_format = workbook.add_format({'bg_color': '#98fb98'})

            # add formatting
            row_num = 1
            num_ests = len(TTXS)
            optimality_counts = np.zeros((3, num_ests)) # rows: bias, var, mse
            for _, row in sim_res_df.iterrows(): 
                bias_cols, var_cols, mse_cols = [], [], []
                for col in sim_res_df.columns:
                    if 'Bias' in col: 
                        bias_cols.append(col)
                    elif 'Variance' in col: 
                        var_cols.append(col)
                    elif 'MSE' in col: 
                        mse_cols.append(col)

                # get optimal indices for bias 
                bias_vals = np.abs(row[bias_cols].values)
                bias_col_idx = np.where(bias_vals == bias_vals.min())[0]
                optimality_counts[0, bias_col_idx] += 1
                bias_col_idx = list((bias_col_idx * 3) + 1)

                # get optimal indices for variance 
                var_vals = row[var_cols].values
                var_col_idx = np.where(var_vals == var_vals.min())[0]
                optimality_counts[1, var_col_idx] += 1
                var_col_idx = list((var_col_idx * 3) + 2)

                # get optimal indices for MSE
                mse_vals = row[mse_cols].values
                mse_col_idx = np.where(mse_vals == mse_vals.min())[0]
                optimality_counts[2, mse_col_idx] += 1
                mse_col_idx = list((mse_col_idx * 3) + 3)

                # add styles 
                for col in bias_col_idx + var_col_idx + mse_col_idx:
                    start_row = row_num
                    start_col = col
                    end_row = start_row
                    end_cold = start_col
                    worksheet.conditional_format(start_row, start_col, end_row, \
                        end_cold, {
                        'type': 'no_errors',
                        'format':   opt_format
                    })
                row_num += 1
        
            # write optimality counts to Excel 
            optimality_counts_df = pd.DataFrame(optimality_counts, index=['Bias', \
                'Variance', 'MSE'], columns=cols)
            optimality_counts_df.to_excel(writer, sheet_name='Optimality Counts')

            # add formatting 
            row_num = 1
            worksheet = writer.sheets['Optimality Counts']
            for _, row in optimality_counts_df.iterrows(): 
                col = np.argmax(row.values) + 1
                start_row = row_num
                start_col = col
                end_row = start_row
                end_cold = start_col
                worksheet.conditional_format(start_row, start_col, end_row, \
                    end_cold, {
                    'type': 'no_errors',
                    'format':   opt_format
                })
                row_num += 1
            print('Wrote simulation results and optimality counts to Excel')
    except Exception as e: 
        print('Unable to write simulation results and optimality counts to '
            + 'Excel. Exception details: ', e)

def run_rate_tests_proc1(): 
    test_cases = TEST_CASES_PROC1
    save_file_name = 'est_rate_proc1.xlsx'
    ttx = False 
    _run_sims(test_cases, save_file_name, ttx)

def run_ttx_tests_proc1(): 
    test_cases = TEST_CASES_PROC1
    save_file_name = 'est_ttx_proc1.xlsx'
    ttx = True 
    _run_sims(test_cases, save_file_name, ttx)

def run_rate_tests_proc2(): 
    test_cases = TEST_CASES_PROC2
    save_file_name = 'est_rate_proc2.xlsx'
    ttx = False 
    _run_sims(test_cases, save_file_name, ttx)

def run_ttx_tests_proc2(): 
    test_cases = TEST_CASES_PROC2
    save_file_name = 'est_ttx_proc2.xlsx'
    ttx = True 
    _run_sims(test_cases, save_file_name, ttx)

def run_rate_tests_proc3(): 
    test_cases = TEST_CASES_PROC3
    save_file_name = 'est_rate_proc3.xlsx'
    ttx = False 
    _run_sims(test_cases, save_file_name, ttx)

def run_ttx_tests_proc3(): 
    test_cases = TEST_CASES_PROC3
    save_file_name = 'est_ttx_proc3.xlsx'
    ttx = True 
    _run_sims(test_cases, save_file_name, ttx)

def run_rate_tests_proc4(): 
    test_cases = TEST_CASES_PROC4
    save_file_name = 'est_rate_proc4.xlsx'
    ttx = False 
    _run_sims(test_cases, save_file_name, ttx)

def run_ttx_tests_proc4(): 
    test_cases = TEST_CASES_PROC4
    save_file_name = 'est_ttx_proc4.xlsx'
    ttx = True 
    _run_sims(test_cases, save_file_name, ttx)

if __name__=='__main__': 
    # start timer
    start = time.time()

    # process 1 
    run_rate_tests_proc1()
    run_ttx_tests_proc1()

    # process 2
    run_rate_tests_proc2()
    run_ttx_tests_proc2()

    # process 3
    run_rate_tests_proc3()
    run_ttx_tests_proc3()

    # # process 4
    run_rate_tests_proc4()
    run_ttx_tests_proc4()

    # end timer, print time elapsed 
    end = time.time()
    print(f'Time Elapsed: {(end - start)/3600} hours')