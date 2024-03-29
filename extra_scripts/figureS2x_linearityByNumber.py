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

def build_sim(n_fishes = 5,params=Params()):
    params.outcome_params = [0.5,-0.5,-0.99]
    params.set_params()
    params.n_fish = n_fishes 
    params.size=50

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
    params.n_rounds = 15 
    params.iterations = 200
    return params,sim
## Set up a tank
      
def run_sim(params,window=3):
    fishes = [Fish(f,params) for f in range(params.n_fish)]
    tank = Tank(fishes,params)

    n_windows = tank.n_rounds - window + 1

    lin_array = np.zeros([params.iterations,n_windows])
    for i in tqdm(range(params.iterations)):
        fishes = [Fish(f,params) for f in range(params.n_fish)]
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
    tank_sizes = [5,50]
    fig,ax = plt.subplots()

    cmap = plt.cm.get_cmap('viridis')

    for n in range(len(tank_sizes)):
        n_fishes = tank_sizes[n]
        params,sim = build_sim(n_fishes)
        lin_array,n_windows = run_sim(params,window=6)
        lins.append(lin_array)

        xs = np.arange(n_windows)
        mean_lin = np.nanmean(lin_array[:,:n_windows],axis=0)
        sem_lin = np.nanstd(lin_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)

        ax.plot(xs,mean_lin,color='black')
        ax.fill_between(xs,mean_lin - sem_lin, mean_lin + sem_lin,alpha=0.5,color=cmap(1-n/len(tank_sizes)),label='n_individuals: ' + str(n_fishes))

    ax.set_xlabel('n rounds of contests')
    ax.set_ylabel('Linearity (1-triad ratio)')
    ax.set_title('Linearity changes with more individuals')
    ax.legend()
    fig.savefig('./figures/figS2x_linearityByNumer.png',dpi=300)
    PLOT = True
    if PLOT:
        plt.show()
