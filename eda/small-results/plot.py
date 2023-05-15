import re
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as st

DATA_PATH = "./results/"
K_s = [2,4,6,8,10,12,14,16]
SIZES = [70,80]
RATIO = 1
#SEEDS = [10582, 15026, 1669, 18874, 21247, 31115, 31720, 6471, 7751, 8220]
SEEDS=None

def get_data(size, k, only_base_pi, depot, seeds=None):
    total_costs = []
    wait_moves = []
    exploration_moves = []

    n = 0
    for filename in os.listdir(DATA_PATH):
        params = filename.split('_')
        if ((seeds is None) and (params[0] == str(size)) and
                params[2] == str(k) and
                params[9] == str(only_base_pi) and
                params[10] == str(depot)):
            result = pd.read_excel(DATA_PATH+filename)
            total_costs.append(result['Total Cost'].values[0])
            wait_moves.append(result['Wait Cost'].values[0])
            exploration_moves.append(result['# of Exploration Steps'].values[0])
        if seeds is not None:
            if (params[0] == str(size) and
                params[2] == str(k) and
                params[9] == str(only_base_pi) and
                params[10] == str(depot) and
                int(params[6]) in seeds):
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

def plot_depot_no_exp(size, ratio, dmar_data, bp_data, k_vals, dmar_exp=None, bp_exp=None):
    plt.figure()
    plt.plot(K_s,np.asarray(dmar_data)-np.asarray(dmar_exp),label="DMAR-DEPOT")
    plt.plot(K_s,np.asarray(bp_data)-np.asarray(bp_exp),label="BP-DEPOT")
    if ratio == 1:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 2:1")
    elif ratio == 2:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 1:1")
    elif ratio == 3:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 1:2")
    plt.xlabel("Agent View Radius (k)")
    plt.ylabel("Number of Moves (No-Exploration included)")
    plt.legend()
    plt.xticks(ticks=k_vals)
    plt.savefig(f"{size}-BP-DEPOT_vs_DMAR-DEPOT_no-exp.png")

def plot_depot(size, ratio, dmar_data, bp_data, k_vals,
                low_dmar_CIs, high_dmar_CIs,
                low_bp_CIs, high_bp_CIs, dmar_exp=None, bp_exp=None):
    plt.figure()
    plt.plot(K_s,dmar_data,label="DMAR-DEPOT")
    plt.plot(K_s,bp_data,label="BP-DEPOT")
    plt.fill_between(k_vals, low_dmar_CIs, high_dmar_CIs, color='b', alpha=0.1)
    plt.fill_between(k_vals, low_bp_CIs, high_bp_CIs, color='r', alpha=0.1)
    plt.plot(k_vals, dmar_exp, '--', label="EXP-MOVES")
    """
    for i,k in enumerate(k_vals):
        text = str(round(dmar_exp[i]/dmar_data[i],2))+","+\
                str(round(bp_exp[i]/bp_data[i],2))
        plt.annotate(text,(k,dmar_data[i]), fontsize=7)
    """
    if ratio == 1:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 2:1")
    elif ratio == 2:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 1:1")
    elif ratio == 3:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 1:2")
    plt.xlabel("Agent View Radius (k)")
    plt.ylabel("Total Number of Moves")
    plt.xticks(ticks=k_vals)
    plt.legend()
    plt.savefig(f"{size}-BP-DEPOT_vs_DMAR-DEPOT.png")

def plot_basic(size, ratio, dmar_data, bp_data, k_vals,
                low_dmar_CIs, high_dmar_CIs,
                low_bp_CIs, high_bp_CIs, dmar_exp=None, bp_exp=None):
    plt.figure()
    plt.plot(K_s, dmar_data, 'b', label="DMAR")
    plt.plot(K_s, bp_data, 'r', label="BP")
    plt.plot(k_vals, dmar_exp,'b--', label="DMAR-EXP-MOVES")
    plt.plot(k_vals, bp_exp,'r--', label="BP-EXP-MOVES")
    plt.fill_between(k_vals, low_dmar_CIs, high_dmar_CIs, color='b', alpha=0.1)
    plt.fill_between(k_vals, low_bp_CIs, high_bp_CIs, color='r', alpha=0.1)
    """
    for i,k in enumerate(k_vals):
        text = (str(round(dmar_exp[i]/dmar_data[i],2))+","+\
                    str(round(bp_exp[i]/bp_data[i],2)))
        print(k)
        plt.annotate(text, (k,0), fontsize=7)
    """
    if ratio == 1:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 2:1")
    elif ratio == 2:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 1:1")
    elif ratio == 3:
        plt.title(f"{size}x{size} w/ Agent-Task ratio of 1:2")
    plt.xlabel("Agent View Radius (k)")
    plt.ylabel("Total Number of Moves")
    plt.legend()
    plt.xticks(ticks=k_vals)
    plt.savefig(f"{size}-BP_vs_DMAR.png")

for size in SIZES:
    plt.figure()
    dmar_costs, dmar_exp_costs = [], []
    bp_costs, bp_exp_costs = [], []
    dmar_depot_costs, dmar_depot_exp_costs = [], []
    bp_depot_costs, bp_depot_exp_costs = [], []

    low_dmar_CIs, high_dmar_CIs = [], []
    low_bp_CIs, high_bp_CIs = [], []
    low_dmar_depot_CIs, high_dmar_depot_CIs = [], []
    low_bp_depot_CIs, high_bp_depot_CIs = [], []

    for k in K_s:
        print(k)
        dmar_CI, dmar_cost, _, dmar_exp_cost = get_data(size,k,False,False,SEEDS)
        bp_CI, bp_cost, _, bp_exp_cost = get_data(size,k,True,False,SEEDS)
        dmar_depot_CI, dmar_depot_cost, _, dmar_depot_exp_cost = get_data(size,k,False,True,SEEDS)
        bp_depot_CI, bp_depot_cost, _, bp_depot_exp_cost = get_data(size,k,True,True,SEEDS)

        dmar_costs.append(dmar_cost)
        dmar_exp_costs.append(dmar_exp_cost)
        bp_costs.append(bp_cost)
        bp_exp_costs.append(bp_exp_cost)
        dmar_depot_costs.append(dmar_depot_cost)
        dmar_depot_exp_costs.append(dmar_depot_exp_cost)
        bp_depot_costs.append(bp_depot_cost)
        bp_depot_exp_costs.append(bp_depot_exp_cost)

        low_dmar_CIs.append(dmar_CI[0])
        high_dmar_CIs.append(dmar_CI[1])
        low_bp_CIs.append(bp_CI[0])
        high_bp_CIs.append(bp_CI[1])
        low_dmar_depot_CIs.append(dmar_depot_CI[0])
        high_dmar_depot_CIs.append(dmar_depot_CI[1])
        low_bp_depot_CIs.append(bp_depot_CI[0])
        high_bp_depot_CIs.append(bp_depot_CI[1])

        #plt.annotate(str(round(dmar_depot_exp_cost/dmar_depot_cost, 2)*100)+"%",
        #             (k, dmar_depot_cost))
        #plt.annotate(str(round(bp_depot_exp_cost/bp_depot_cost, 2)*100)+"%",
        #             (k, bp_depot_cost))

    print(dmar_costs, bp_costs)
    plot_basic(size,RATIO,dmar_costs,bp_costs,K_s,
                low_dmar_CIs,high_dmar_CIs,low_bp_CIs,high_bp_CIs,
                dmar_exp_costs,bp_exp_costs)
    plot_depot(size,RATIO,dmar_depot_costs,bp_depot_costs,K_s,
                low_dmar_depot_CIs,high_dmar_depot_CIs,
                low_bp_depot_CIs,high_bp_depot_CIs,
                dmar_depot_exp_costs,bp_depot_exp_costs)
    plot_depot_no_exp(size,RATIO,dmar_depot_costs,bp_depot_costs,K_s,
                        dmar_depot_exp_costs, bp_depot_exp_costs)
    """
    plt.fill_between(K_s, low_dmar_depot_CIs, high_dmar_depot_CIs, color='b', alpha=0.1)
    plt.fill_between(K_s, low_bp_depot_CIs, high_bp_depot_CIs, color='r', alpha=0.1)

    #plt.fill_between(K_s, low_dmar_CIs, high_dmar_CIs, color='b', alpha=0.1)
    #plt.fill_between(K_s, low_bp_CIs, high_bp_CIs, color='r', alpha=0.1)

    plt.plot(K_s,dmar_depot_costs,label="DMAR-DEPOT")
    plt.plot(K_s,bp_depot_costs,label="BP-DEPOT")
    #plt.plot(K_s,np.asarray(dmar_depot_costs)-np.asarray(dmar_depot_exp_costs),label="DMAR-DEPOT")
    #plt.plot(K_s,dmar_exp_costs,label="DMAR-EXP")
    #plt.plot(K_s,dmar_costs,label="DMAR")
    #plt.plot(K_s,np.asarray(bp_depot_costs)-np.asarray(bp_depot_exp_costs),label="BP-DEPOT")
    #plt.plot(K_s,bp_exp_costs,label="BP-EXP")
    #plt.plot(K_s,bp_costs,label="BP")
    plt.title(f"{size}x{size} size with Agent:Task ratio of 2:1")
    plt.legend()
    plt.xticks(ticks=K_s)
    plt.xlabel("Agent View Radius")
    plt.ylabel("Total Number of Moves")
    #plt.savefig(f"{size}-BP_vs_DMAR.png")
    #plt.savefig(f"{size}-BP-DEPOT_vs_DMAR-DEPOT.png")
    #plt.savefig(f"{size}-BP-DEPOT_vs_DMAR-DEPOT_no-exp.png")
    """
