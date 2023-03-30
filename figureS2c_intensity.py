#! /usr/bin/env python

## Trying something new:
""" 
This script contains all the code for plotting figure 1 of Ammon's BayesBattleBots paper
For questions, contact Ammon Perkes (perkes.ammon@gmail.com)
"""

import numpy as np
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from matplotlib import cm
from params import Params

## Define some global variables to determine if you will plot and save figures.
PLOT = True
SAVE = False

def build_sim(sd_size=2,params=Params()):
    params.outcome_params = [0.5,-0.5,-.99]
    params.set_params()
    if sd_size == 0:
        params.size=50
    else:
        params.size=None
        params.mean_size = 50
        params.sd_size=sd_size

    params.effort_method = 'SmoothPoly'
    params.poly_param_c = 0
    params.awareness = 15

    params.acuity = 2
    params.post_acuity = True
    params.f_method = 'shuffled'

    params.n_fish = 5
    #params.n_fights = 30
    params.iterations = 500
    params.n_rounds = 30
    return params

def run_sim(params):
## Set up a tank
    cost_array = np.empty([params.iterations,params.n_fish,params.n_rounds * (params.n_fish -1)])
    for i in tqdm(range(params.iterations)):
        fishes = [Fish(f,params) for f in range(5)]

        tank = Tank(fishes,params)
        tank.run_all(progress=False)

        for f in range(len(fishes)):
            cost_array[i,f] = np.array(tank.fishes[f].win_record)[:,3]
    return cost_array

if __name__ == '__main__':
    intens = []
    fig,ax = plt.subplots()
    stds = [0,2,4,8]

    cmap = plt.cm.get_cmap('viridis')
    
    for s in range(len(stds)):
        std = stds[s]
        params = build_sim(std)
        cost_array = run_sim(params)

        mean_int = np.nanmean(cost_array,axis=(0,1))
        sem_int = np.std(cost_array,axis=(0,1)) / np.sqrt(params.iterations)
        xs = np.arange(len(mean_int))
        ax.plot(xs,mean_int,color='black')
        ax.fill_between(xs,mean_int-sem_int,mean_int+sem_int,color=cmap(1-s/len(stds)),label='std: ' + str(s),alpha=0.5)

    ax.set_xlabel('Contest number')
    ax.set_ylabel('Mean fight Intensity\n(+/- SEM of iterations')
    ax.set_title('Fight intensity decreases over repeated contests')

    fig.savefig('./figures/figS2c_intensity.png',dpi=300)

    if PLOT:
        plt.show()
