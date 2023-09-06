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
from simulation import Simulation

## Define some global variables to determine if you will plot and save figures.
PLOT = True
SAVE = False

sim = Simulation()
params = Params()
params.outcome_params = [0.5,-0.5,-0.8]
#params.set_L()
params.awareness = 0.5
params.acuity = 0.1

params.set_params()
#params.size=50
#params.prior = True

#params.effort_method = [None,'!']
#params.effort_method = 'ExplorePoly'
params.effort_method = 'SmoothPoly'
params.poly_param_c = 0

params.post_acuity = True
params.f_method = 'shuffled'

params.n_fish = 5
params.n_rounds = 30
params.iterations = 500
#params.n_rounds = params.n_fights * (params.n_fish-1)

## Set up a tank
cost_array = np.empty([params.iterations,params.n_fish,params.n_rounds * (params.n_fish-1)])
error_array = np.empty([params.iterations,params.n_fish,params.n_rounds * (params.n_fish - 1) + 1])

for i in tqdm(range(params.iterations)):
    fishes = [Fish(f,params) for f in range(5)]

    tank = Tank(fishes,params)
    tank.run_all(progress=False)

    error_array[i] = sim._calc_error(tank)
    for f in range(len(fishes)):
        cost_array[i,f] = np.array(tank.fishes[f].win_record)[:,3]
    

fig,(ax,ax1) = plt.subplots(2)

mean_int = np.nanmean(cost_array,axis=(0,1))
sem_int = np.nanstd(cost_array,axis=(0,1)) / np.sqrt(params.iterations)
ax.plot(mean_int,color='black')
ax.fill_between(range(len(mean_int)),mean_int-sem_int,mean_int+sem_int,color='gray',alpha=0.5)
ax.set_xlabel('Contest number')
ax.set_ylabel('Mean fight Intensity\n(+/- SEM of iterations')
ax.set_title('Fight intensity decreases over repeated contests')


## Plot error
mean_err = np.mean(error_array,axis=(0,1))
sem_err = np.std(error_array,axis=(0,1)) / np.sqrt(params.iterations)
ax1.plot(mean_err,color='black')
ax1.fill_between(range(len(mean_err)),mean_err-sem_err,mean_err+sem_err,color='gray',alpha=0.5)
ax1.set_xlabel('Contest number')
ax1.set_ylabel('Mean error \n(+/- SEM of iterations')
ax1.set_title('Estimate error decreases over repeated contests')

#fig.savefig('./figures/figure2c_intensity.png',dpi=300)

if PLOT:
    plt.show()