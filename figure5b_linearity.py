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
params.outcome_params = [0.5,-0.5,-0.7]
params.set_params()
if True:
    params.size=50
else:
    params.size=None
    params.mean_size = 50
    params.sd_size = 2

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
#params.n_fights = 40
params.n_rounds = 30
params.iterations = 500
## Set up a tank
  
## Set up dummy tank to make calculating rounds easier
fishes = [Fish(f,params) for f in range(5)]
tank = Tank(fishes,params)

window = 6
n_windows = tank.n_rounds - window + 1

lin_array = np.zeros([params.iterations,n_windows])
#import pdb;pdb.set_trace()
for i in tqdm(range(params.iterations)):
    fishes = [Fish(f,params) for f in range(5)]
    tank = Tank(fishes,params)
    tank.run_all(print_me=False,progress=False)
    lin_list = []
    for w in range(n_windows):
        #idx = slice(int(w*window/2),int(w*window/2 + window))
        idx = slice(w,w+window)
        linearity,[d,p] = sim._calc_linearity(tank,idx)
        lin_array[i,w] = linearity
        #print(linearity)
fig,ax = plt.subplots()
print(n_windows,len(tank.history))
xs = np.arange(n_windows)
mean_lin = np.nanmean(lin_array[:,:n_windows],axis=0)
sem_lin = np.nanstd(lin_array[:,:n_windows],axis=0) / np.sqrt(params.iterations)
ax.plot(xs,mean_lin,color='black')
ax.fill_between(xs,mean_lin - sem_lin, mean_lin + sem_lin,alpha=0.5,color='gray')
ax.set_xlabel('n rounds of contests')
ax.set_ylabel('Linearity (1-triad ratio)')
ax.set_title('Linearity increases over time')

fig.savefig('./figures/figure2b_linearity.png',dpi=300)
PLOT = True
if PLOT:
    plt.show()
