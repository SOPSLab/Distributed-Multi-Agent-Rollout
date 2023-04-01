import re
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as st

DATA_PATH = "./results/"
K_s = [2,3,4,5,6]
SIZES = [10,20,30,40]

def get_data(size, k, only_base_pi, depot):
    total_costs = []
    wait_moves = []
    exploration_moves = []

    n = 0
    for filename in os.listdir(DATA_PATH):
        params = filename.split('_')
        if ((params[0] == str(size)) and
                params[2] == str(k) and
                params[9] == str(only_base_pi) and
                params[10] == str(depot)):
            result = pd.read_excel(DATA_PATH+filename)
            total_costs.append(result['Total Cost'].values[0])
            wait_moves.append(result['Wait Cost'].values[0])
            exploration_moves.append(result['# of Exploration Steps'].values[0])

    confidence_interval = st.t.interval(alpha=0.95,
                                        df=len(total_costs)-1,
                                        loc=np.mean(total_costs),
                                        scale=st.sem(total_costs))

    return confidence_interval, \
            np.mean(total_costs), \
            np.mean(wait_moves), \
            np.mean(exploration_moves)

for size in SIZES:
    plt.figure()
    dmar_costs, dmar_exp_costs = [], []
    bp_costs, bp_exp_costs = [], []
    low_dmar_CIs, high_dmar_CIs = [], []
    low_bp_CIs, high_bp_CIs = [], []
    for k in K_s:
        dmar_CI, dmar_cost, _, dmar_exp_cost = get_data(size,k,False,False)
        bp_CI, bp_cost, _, bp_exp_cost = get_data(size,k,True,False)
        dmar_costs.append(dmar_cost)
        dmar_exp_costs.append(dmar_exp_cost)
        bp_costs.append(bp_cost)
        bp_exp_costs.append(bp_exp_cost)
        low_dmar_CIs.append(dmar_CI[0])
        high_dmar_CIs.append(dmar_CI[1])
        low_bp_CIs.append(bp_CI[0])
        high_bp_CIs.append(bp_CI[1])

        plt.annotate(str(round(dmar_exp_cost/dmar_cost, 2)*100)+"%",
                     (k, dmar_cost))
        plt.annotate(str(round(bp_exp_cost/bp_cost, 2)*100)+"%",
                     (k, bp_cost))

    print(dmar_costs, bp_costs)
    plt.fill_between(K_s, low_dmar_CIs, high_dmar_CIs, color='b', alpha=0.1)
    plt.fill_between(K_s, low_bp_CIs, high_bp_CIs, color='r', alpha=0.1)
    plt.plot(K_s,dmar_costs,label="DMAR")
    #plt.plot(K_s,dmar_exp_costs,label="DMAR-EXP")
    plt.plot(K_s,bp_costs,label="BP")
    #plt.plot(K_s,bp_exp_costs,label="BP-EXP")
    plt.title(f"BP v/s DMAR for {size}x{size} grid")
    plt.legend()
    plt.xticks(ticks=K_s)
    plt.savefig(f"{size}-BP_vs_DMAR.png")
