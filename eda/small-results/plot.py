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

bp_dmar_10 = plt.figure()
bp_dmar_20 = plt.figure()
bp_dmar_30 = plt.figure()
bp_dmar_40 = plt.figure()

for size in SIZES:
    plt.figure()
    dmar_costs = []
    bp_costs = []
    for k in K_s:
        cf_is, dmar_cost, _, _ = get_data(size,k,False,False)
        cf_is, bp_cost, _, _ = get_data(size,k,True,False)
        dmar_costs.append(dmar_cost)
        bp_costs.append(bp_cost)
    print(dmar_costs, bp_costs)
    plt.plot(K_s,dmar_costs,label="DMAR")
    plt.plot(K_s,bp_costs,label="BP")
    plt.title(f"BP v/s DMAR for {size}x{size} grid")
    plt.legend()
    plt.xticks(ticks=K_s)
    plt.savefig(f"{size}-BP_vs_DMAR.png")
print(get_data(40,3,True,False))
print(get_data(40,3,False,False))

print(get_data(40,6,True,False))
print(get_data(40,6,False,False))

print(get_data(10,2,True,False))
print(get_data(10,2,False,False))

print(get_data(10,6,True,False))
print(get_data(10,6,False,False))
