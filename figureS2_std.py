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
    params.outcome_params = [0.5,-0.5,-0.9]
    params.n_fish = 5
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
    params.awareness = 15

    params.acuity = 10
    params.post_acuity = True
    params.f_method = 'shuffled'
    #params.n_fights = 40
    params.n_rounds = 30
    params.iterations = 500
 
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

    stds = [0,2,4,8]

    cmap = plt.cm.get_cmap('viridis')
    for s in range(len(stds)):
        std = stds[s]
        sim, params = build_sim(std)
        lin_array, stab_array, cost_array, n_windows = run_sim(params)
        #stabs.append(stab_array)
        xs = np.arange(n_windows)

        mean_stab = np.nanmean(stab_array[:,:n_windows],axis=0)
        sem_stab = np.nanstd(stab_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)

        mean_lin = np.nanmean(lin_array[:,:n_windows],axis=0)
        sem_lin = np.nanstd(lin_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)

        mean_int = np.nanmean(cost_array,axis=(0,1))
        sem_int = np.std(cost_array,axis=(0,1)) / np.sqrt(params.iterations) 

        ax.plot(xs,mean_stab,color='black')
        ax.fill_between(xs,mean_stab - sem_stab, mean_stab + sem_stab,alpha=0.5,color=cmap(1- s/len(stds)),label='std: ' + str(std))

        ax1.plot(xs,mean_lin,color='black')
        ax1.fill_between(xs,mean_lin - sem_lin, mean_lin + sem_lin,alpha=0.5,color=cmap(1-s/len(stds)),label='Std: ' + str(std))

        xs_int = np.arange(len(mean_int))
        ax2.plot(xs_int,mean_int,color='black')
        ax2.fill_between(xs_int,mean_int-sem_int,mean_int+sem_int,color=cmap(1-s/len(stds)),label='std: ' + str(std),alpha=0.5)


    ax.set_xlabel('n rounds of contests')
    ax.set_ylabel('Stability (prop consistent with mean)')
    ax.set_title('Stability increases over time')
    ax.legend()

    ax1.set_xlabel('n rounds of contests')
    ax1.set_ylabel('Linearity (1-triad ratio)')
    ax1.set_title('Linearity increases over time')
    ax1.legend()

    ax2.set_xlabel('Contest number')
    ax2.set_ylabel('Mean fight Intensity\n(+/- SEM of iterations')
    ax2.set_title('Fight intensity decreases over repeated contests')

    if False:
        fig.savefig('./figures/figS2a_stability.png',dpi=300)
        fig1.savefig('./figures/figS2b_linearity.png',dpi=300)
        fig2.savefig('./figures/figS2c_intensity.png',dpi=300)

    if True:
        plt.show()

