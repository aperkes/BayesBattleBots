#! /usr/bin/env python

## Trying something new:
""" 
This script contains all the code for plotting figure 1 of Ammon's BayesBattleBots paper
For questions, contact Ammon Perkes (perkes.ammon@gmail.com)
"""

import numpy as np
from matplotlib import pyplot as plt
import copy

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from matplotlib import cm
from params import Params
from tqdm import tqdm

## Define some global variables to determine if you will plot and save figures.
PLOT = True
SAVE = False

## Look, a smarter person would have made one function that could work in all these
## Maybe I'll get there, but it's actually a lot more manageable this way, really

def build_sim(s,f,l,params=Params()):
    params.outcome_params = [s,f,l]
    params.n_fish = 5
    params.set_params()

    sim = Simulation()

    params.effort_method = 'SmoothPoly'
    params.poly_param_c = 0
    params.awareness = 5

    params.acuity = 5 
    params.post_acuity = True
    params.f_method = 'shuffled'
    #params.n_fights = 40
    params.n_rounds = 20
    params.iterations = 200
 
    return sim, params

def run_sim(params,window=3):
    n_windows = params.n_rounds - window + 1

    stab_array = np.zeros([params.iterations,n_windows])
    lin_array = np.zeros([params.iterations,n_windows])
    cost_array = np.empty([params.iterations,params.n_fish,params.n_rounds * (params.n_fish -1)])

    for i in tqdm(range(params.iterations)):
        fishes = [Fish(f,params) for f in range(params.n_fish)]
        tank = Tank(fishes,params)
        tank.run_all(print_me=False,progress=False)
        lin_list = []
        for w in range(n_windows):
            idx = slice(w,w+window)
            stability,_ = sim._calc_stability(tank,idx)
            linearity,[d,p] = sim._calc_linearity(tank,idx)
            lin_array[i,w] = linearity
            stab_array[i,w] = stability
        for f in range(len(fishes)):
            cost_array[i,f] = np.array(tank.fishes[f].win_record)[:,3]
    return lin_array, stab_array, cost_array, n_windows

if __name__ == '__main__':
    stabs = []
    fig,ax = plt.subplots() ## Stability
    fig1,ax1 = plt.subplots() ## Linearity
    fig2,ax2 = plt.subplots() ## Cost of fights (intensity, efficiency)

    s_list = [-1,-0.9,-0.8,-0.2,-0.1,0,0.1,0.2,0.8,0.9,1]
    f_list = [-1,-0.9,-0.8,-0.2,-0.1,0,0.1,0.2,0.8,0.9,1]
    l_list = [-1,-0.9,-0.8,-0.2,-0.1,0,0.1,0.2,0.8,0.9,1]

    stab_tern_0 = np.zeros([len(s_list),len(f_list),len(l_list)])
    lin_tern_0 = np.zeros_like(stab_tern_0)
    int_tern_0 = np.zeros_like(stab_tern_0)

    stab_tern_20 = np.zeros_like(stab_tern_0)
    lin_tern_20 = np.zeros_like(stab_tern_0)
    int_tern_20 = np.zeros_like(stab_tern_0)

    cmap = plt.cm.get_cmap('viridis')
    for s_ in range(len(s_list)):
        s = s_list[s_]
        for f_ in range(len(f_list)):
            f = f_list[f_]
            for l_ in range(len(l_list)):
                l = f_list[l_]
                print(s,f,l)

                sim, params = build_sim(s,f,l)
                lin_array, stab_array, cost_array, n_windows = run_sim(params)
                lin_0,lin_20 = np.nanmean(lin_array[0]),np.nanmean(lin_array[-1])
                stab_0,stab_20 = np.nanmean(stab_array[0]),np.nanmean(stab_array[-1])
                int_0,int_20 = np.nanmean(cost_array[0]),np.nanmean(cost_array[-1])
                
                lin_tern_0[s_,f_,l_] = lin_0 
                lin_tern_20[s_,f_,l_] = lin_20 
                
                stab_tern_0[s_,f_,l_] = stab_0 
                stab_tern_20[s_,f_,l_] = stab_20 
                
                int_tern_0[s_,f_,l_] = int_0 
                int_tern_20[s_,f_,l_] = int_20 

    np.save('./results/lin_tern_0.npy',lin_tern_0)
    np.save('./results/lin_tern_20.npy',lin_tern_20)
    np.save('./results/stab_tern_0.npy',stab_tern_0)
    np.save('./results/stab_tern_20.npy',stab_tern_20)
    np.save('./results/int_tern_0.npy',int_tern_0)
    np.save('./results/int_tern_20.npy',int_tern_20)

#import pdb;pdb.set_trace()
print('done!')
