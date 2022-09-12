# import libraries 
from sim_life import cmp_ests
import numpy as np
import pandas as pd
import os
import platform


# specify test cases (s, e, lmb, mu)
TEST_CASES_PROC4 = [
    # later in time, high mu (to simulate high TTRs)
    (1000, 1100, 20, 80),
    (1000, 1100, 20, 100),
    (1000, 1100, 20, 120),
    (1000, 1100, 20, 150),
    (1000, 1100, 20, 200),
    (1000, 1100, 50, 80),
    (1000, 1100, 50, 100),
    (1000, 1100, 50, 120),
    (1000, 1100, 50, 150),
    (1000, 1100, 50, 200),
    (1000, 1100, 80, 80),
    (1000, 1100, 80, 100),
    (1000, 1100, 80, 120),
    (1000, 1100, 80, 150),
    (1000, 1100, 80, 200),
    
    # later in time, high lmb (simulate low rate of IcM creation)
    (1000, 1100, 80, 20),
    (1000, 1100, 100, 20),
    (1000, 1100, 120, 20),
    (1000, 1100, 150, 20),
    (1000, 1100, 200, 20),
    (1000, 1100, 80, 50),
    (1000, 1100, 100, 50),
    (1000, 1100, 120, 50),
    (1000, 1100, 150, 50),
    (1000, 1100, 200, 50),
    (1000, 1100, 80, 80),
    (1000, 1100, 100, 80),
    (1000, 1100, 120, 80),
    (1000, 1100, 150, 80),
    (1000, 1100, 200, 80),
]


def run_tests_proc4():
    # initialize distribution information map 
    sim_dist_info = []
    for s, e, lmb, mu in TEST_CASES_PROC4: 
        res = cmp_ests(s, e, lmb, mu)
        if type(res) == np.ndarray: 
            res = list(res.flatten())
        sim_dist_info.append(res)
    
    # save distribution information as Excel file 
    columns = [
        'E1 E(TTR)',
        'E1 Var(TTR)',
        'E1 MSE',
        'E2 E(TTR)',
        'E2 Var(TTR)',
        'E2 MSE',
        'E5 E(TTR)',
        'E5 Var(TTR)',
        'E5 MSE',
    ]
    str_test_cases = map(lambda x: str(x), TEST_CASES_PROC4)
    dist_info_df = pd.DataFrame(data=sim_dist_info, index=str_test_cases,\
        columns=columns)
    dist_info_df.index.rename('Test case (s, e, lmb, mu)', inplace=True)

    # write to excel sheet
    dir = os.path.dirname(os.path.abspath(__file__))
    plat = platform.platform()
    save_path = dir + '/est_ttr_proc4.xlsx'
    if 'windows' in plat.lower(): 
        save_path = dir + '\\est_ttr_proc4.xlsx'
    try: 
        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            # write results to sheet
            dist_info_df.to_excel(writer, sheet_name='Process 4 TTR estimations')
            workbook = writer.book
            worksheet = writer.sheets['Process 4 TTR estimations']
            opt_format = workbook.add_format({'bg_color': '#98fb98'})

            # add formatting
            row_num = 1
            num_ests = len(sim_dist_info[0]) // 3
            optimality_counts = np.zeros((3, num_ests)) # rows: mean, var, mse
            for idx, row in dist_info_df.iterrows(): 
                vals = idx.split(',')
                mu = int(vals[-1].strip()[:-1])
                e_ttr_cols, var_ttr_cols, mse_ttr_cols = [], [], []
                for col in dist_info_df.columns:
                    if 'E(TTR)' in col: 
                        e_ttr_cols.append(col)
                    elif 'Var(TTR)' in col: 
                        var_ttr_cols.append(col)
                    elif 'MSE' in col: 
                        mse_ttr_cols.append(col)
                e_ttr_vals = np.abs(row[e_ttr_cols].values - mu)
                e_ttr_col_idx = np.argmin(e_ttr_vals)
                optimality_counts[0, e_ttr_col_idx] += 1
                e_ttr_col_idx = (e_ttr_col_idx * 3) + 1
                var_ttr_col_idx = np.argmin(row[var_ttr_cols].values)
                optimality_counts[1, var_ttr_col_idx] += 1
                var_ttr_col_idx = (var_ttr_col_idx * 3) + 2
                mse_ttr_col_idx = np.argmin(row[mse_ttr_cols].values)
                optimality_counts[2, mse_ttr_col_idx] += 1
                mse_ttr_col_idx = (mse_ttr_col_idx * 3) + 3
                for col in [e_ttr_col_idx, var_ttr_col_idx, mse_ttr_col_idx]:
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
            optimality_counts_df = pd.DataFrame(optimality_counts, index=['E(TTR)', \
                'Var(TTR)', 'MSE'], columns=['E1', 'E2', 'E5'])
            optimality_counts_df.to_excel(writer, sheet_name='Process 4 '
                + 'Optimality Counts')

            # add formatting 
            row_num = 1
            worksheet = writer.sheets['Process 4 Optimality Counts']
            for idx, row in optimality_counts_df.iterrows(): 
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
            

if __name__=='__main__': 
    run_tests_proc4()