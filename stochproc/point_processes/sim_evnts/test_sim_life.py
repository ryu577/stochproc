# import libraries 
from sim_life1 import cmp_ests
import pandas as pd
import os
import platform

# specify test cases (s, e, lmd, mu)
TEST_CASES = [
    (50, 75, 1, 1), 
    (50, 100, 1, 1), 
    (50, 125, 1, 1), 
    (50, 150, 1, 1), 
    (50, 75, 2, 3),
    (50, 100, 2, 3), 
    (50, 125, 2, 3), 
    (50, 150, 2, 3), 
    (50, 75, 5, 10), 
    (50, 100, 5, 10), 
    (50, 125, 5, 10), 
    (50, 150, 5, 10), 
    (100, 900, 30, 80), 
    (100, 1000, 30, 80), 
    (100, 1100, 30, 80), 
    (100, 1200, 30, 80), 
    (300, 2100, 80, 50), 
    (300, 2300, 80, 50), 
    (300, 2500, 80, 50), 
    (300, 2700, 80, 50), 
]

def run_tests():
    # initialize distribution information map 
    sim_dist_info = []
    for s, e, lmd, mu in TEST_CASES: 
        sim_dist_info.append(cmp_ests(s, e, lmd, mu))
    
    # save distribution information as CSV file 
    columns = [] 
    for i in range(len(sim_dist_info[0]) // 2): 
        columns.append(f'E{i} E(TTR)')
        columns.append(f'E{i} Var(TTR)')
    dist_info_df = pd.DataFrame(data=sim_dist_info, index=TEST_CASES, columns=columns)
    dist_info_df.index.rename('Test case (s, e, lmb, mu)', inplace=True)
    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plat = platform.platform()
    save_path = dir + '/sim_life_est_ttr.csv'
    if 'windows' in plat.lower(): 
        save_path = dir + '\\sim_life_est_ttr.csv'
    dist_info_df.to_csv(save_path)

if __name__=='__main__': 
    run_tests()