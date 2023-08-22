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
    params.outcome_params = [0.5,-0.5,-0.99]
    params.set_params()
    params.n_fish = 5
    if sd_size == 0:
        params.size=50
    else:
        params.size=None
        params.mean_size = 50
        params.sd_size = sd_size

    params.prior = True
    sim = Simulation()

#params.effort_method = [None,'!']
#params.effort_method = 'ExplorePoly'
    params.effort_method = 'SmoothPoly'
    params.poly_param_c = 0
    params.awareness = 15

    params.acuity = 2
    params.post_acuity = True
    params.f_method = 'shuffled'
    params.n_rounds = 30
    params.iterations = 500
    return params,sim
## Set up a tank
      
def run_sim(params,window=3):
    fishes = [Fish(f,params) for f in range(5)]
    tank = Tank(fishes,params)

    n_windows = tank.n_rounds - window + 1

    lin_array = np.zeros([params.iterations,n_windows])
    for i in tqdm(range(params.iterations)):
        fishes = [Fish(f,params) for f in range(5)]
        tank = Tank(fishes,params)
        tank.run_all(print_me=False,progress=False)
        lin_list = []
        for w in range(n_windows):
            idx = slice(w,w+window)
            linearity,[d,p] = sim._calc_linearity(tank,idx)
            lin_array[i,w] = linearity
    return lin_array,n_windows

if __name__ == '__main__':
    lins = []
    stds = [0,2,4,8]
    fig,ax = plt.subplots()

    cmap = plt.cm.get_cmap('viridis')

    styles = ['solid','dotted','dashed','dashdot']
    for s in range(len(stds)):
        std = stds[s]
        params,sim = build_sim(std)
        lin_array,n_windows = run_sim(params,window=6)
        lins.append(lin_array)

        xs = np.arange(n_windows)
        mean_lin = np.nanmean(lin_array[:,:n_windows],axis=0)
        sem_lin = np.nanstd(lin_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)

        ax.plot(xs,mean_lin,color='black',linestyle=styles[s])
        ax.fill_between(xs,mean_lin - sem_lin, mean_lin + sem_lin,alpha=0.5,color=cmap(1-s/len(stds)),label='Std: ' + str(std))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right')

    ax.set_xlabel('n rounds of contests')
    ax.set_ylabel('Linearity (1-triad ratio)')
    ax.set_title('Linearity increases over time')
    #ax.legend()
    fig.savefig('./figures/figS2b_linearity.png',dpi=300)
    PLOT = True
    if PLOT:
        plt.show()
