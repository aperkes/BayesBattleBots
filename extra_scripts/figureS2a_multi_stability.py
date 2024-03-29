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

def build_sim(sd_size=2,params=Params()):
    #params.outcome_params = [-0.5,-0.5,-0.7]
    #params.outcome_params = [0.0,0.0,-0.9]
    params.outcome_params = [0.1,0.0,-0.99]
    params.n_fish = 5
    params.awareness = 0.3 
    params.acuity = 0
    params.set_params()

    if sd_size == 0:
        params.size = 50
    else:
        params.size=None
        params.mean_size = 50
        params.sd_size = sd_size

    params.prior = True
    sim = Simulation()

    params.effort_method = 'SmoothPoly'
    params.poly_param_c = 0

    params.post_acuity = True
    params.f_method = 'shuffled'
    #params.n_fights = 40
    params.n_rounds = 30
    params.iterations = 100
 
    return sim, params

def run_sim(params,window=3):
    n_windows = params.n_rounds - window + 1
    stab_array = np.zeros([params.iterations,n_windows])

    for i in tqdm(range(params.iterations)):
        fishes = [Fish(f,params) for f in range(params.n_fish)]
        tank = Tank(fishes,params)
        tank.run_all(print_me=False,progress=False)
        lin_list = []
        for w in range(n_windows):
            idx = slice(w,w+window)
            stability,_ = sim._calc_stability(tank,idx)
            stab_array[i,w] = stability

    return stab_array,n_windows

if __name__ == '__main__':
    stabs = []
    fig,ax = plt.subplots()
    stds = [0,2,4,8]
    #stds = [0]

    cmap = plt.cm.get_cmap('viridis')
    styles = ['solid','dotted','dashed','dashdot']
    for s in range(len(stds)):
        std = stds[s]
        print(std)
        sim, params = build_sim(std)
        stab_array,n_windows = run_sim(params)
        stabs.append(stab_array)
        xs = np.arange(n_windows)

        mean_stab = np.nanmean(stab_array[:,:n_windows],axis=0)

        sem_stab = np.nanstd(stab_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)
        ax.plot(xs,mean_stab,color='black',linestyle=styles[s])
        ax.fill_between(xs,mean_stab - sem_stab, mean_stab + sem_stab,alpha=0.5,color=cmap(1- s/len(stds)),label='std: ' + str(std))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right')

    ax.set_xlabel('n rounds of contests')
    ax.set_ylabel('Stability (prop consistent with mean)')
    ax.set_title('Stability increases over time')
    #ax.legend()
    fig.savefig('./figures/figS2a_stability.png',dpi=300)

    if True:
        plt.show()

