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

## Define some global variables to determine if you will plot and save figures.
PLOT = True
SAVE = False

params = Params()
params.outcome_params = [0.6,0.3,.1]
params.set_L()
params.size=50
params.prior = True

#params.effort_method = [None,'!']
#params.effort_method = 'ExplorePoly'
params.effort_method = 'SmoothPoly'
params.poly_param_c = 0
params.awareness = 15

params.acuity = 2
params.post_acuity = True
params.f_method = 'shuffled'
params.n_fights = 30

params.iterations = 100

## Set up a tank
for i in range(params.iterations):
    fishes = [Fish(f,params) for f in range(5)]

    tank = Tank(fishes,params)
    tank.run_all()

#effort_array = np.zeros([len(fishes),len(win_record)])
cost_array = np.zeros([len(fishes),len(win_record)])
for f in range(len(fishes)):
    cost_record = np.array(tank.fishes[f].win_record)[:,3]
    #effort_record = np.array(tank.fishes[f].win_record)[:,2]
    #effort_array[f] = effort_record
    cost_array[f] = cost_record

fig2,ax2 = plt.subplots()

smooth_costs = gaussian_filter1d(np.mean(cost_array,0),3)
smooth_effort = gaussian_filter1d(np.mean(effort_array,0),3)
ax2.plot(smooth_costs*4,color='black')
ax2.plot(smooth_effort*4,linestyle=':',color='black')

## Figure 3c: Estimate over time, to show that bayes is accurate

fig3,ax3 = tank.plot_estimates(food=False)

fig4,ax4 = plt.subplots()
f_estimates = [f.estimate for f in fishes]
f_max = np.argmax(f_estimates)
f = f_max
print(f)
ax4.plot(fishes[f].xs,fishes[f].prior)
ax4.axvline(fishes[f].size,color='black')
ax4.axvline(fishes[f].range_record[-1][1])
ax4.axvline(fishes[f].range_record[-1][2])

if PLOT:
    plt.show()
