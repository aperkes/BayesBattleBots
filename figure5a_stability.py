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

params = Params()
params.n_fish = 5
params.outcome_params = [0.7,-0.5,-0.8]
#params.outcome_params = [-0.5,0.5,-0.8]
params.acuity = 0.0
params.awareness = 0.5

params.set_params()


if False:
    params.size=50
else:
    params.size=None
    params.mean_size = 50
    params.sd_size = 8

#params.prior = True
sim = Simulation()

#params.effort_method = [None,'!']
#params.effort_method = 'ExplorePoly'
params.effort_method = 'SmoothPoly'
params.poly_param_c = 0
#params.awareness = 15

#params.acuity = 10
params.post_acuity = True
params.f_method = 'shuffled'
params.n_rounds = 25
#params.n_rounds = 8
params.iterations = 500
## Set up a tank
  
## Set up dummy tank to make calculating rounds easier
fishes = [Fish(f,params) for f in range(params.n_fish)]
tank = Tank(fishes,params)

window = 3
#n_windows = tank.n_rounds * 2 // window - 1
n_windows = tank.n_rounds - window +1
print(tank.n_rounds,params.n_rounds,n_windows,params.n_fish)
stab_array = np.zeros([params.iterations,n_windows])

#import pdb;pdb.set_trace()
for i in tqdm(range(params.iterations)):
    fishes = [Fish(f,params) for f in range(5)]
    tank = Tank(fishes,params)
    tank.run_all(print_me=False,progress=False)
    #import pdb;pdb.set_trace()
    lin_list = []
    for w in range(n_windows):
        #idx = slice(int(w*window/2),int(w*window/2 + window))
        idx = slice(w,w+window)
        stability,_ = sim._calc_stability(tank,idx)
        stab_array[i,w] = stability


fig,ax = plt.subplots()
print(n_windows,len(tank.history))
xs = np.arange(n_windows)
mean_stab = np.nanmean(stab_array[:,:n_windows],axis=0)
sem_stab = np.nanstd(stab_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)
ax.plot(xs,mean_stab,color='black')
ax.fill_between(xs,mean_stab - sem_stab, mean_stab + sem_stab,alpha=0.5,color='gray')
ax.set_xlabel('n rounds of contests')
ax.set_ylabel('Stability (prop consistent with mean)')
ax.set_title('Stability increases over time')

fig.savefig('./figures/figure2a_stability.png',dpi=300)
print(mean_stab,sem_stab)

PLOT = True
if PLOT:
    plt.show()
