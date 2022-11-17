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

## Define some global variables to determine if you will plot and save figures.
PLOT = 1
SAVE = False

params = SimParams()
params.outcome_params = [1,0.3,.1]
params.effort_method = [None,'!']
params.f_method = 'shuffled'
## Set up a tank
if 1:
    fishes = [Fish(f,age=50,size=47,prior=True,insight=True,acuity=0,effort_method=params.effort_method,fight_params=params.outcome_params) for f in range(5)]
else:
    fishes = [Fish(f,insight=True,acuity=0,effort_method=params.effort_method,fight_params=params.outcome_params) for f in range(5)]

tank = Tank(fishes,n_fights = 50,death=False,f_method=params.f_method,f_params=params.outcome_params)
tank.run_all()

## Figure 3a: Proportion of fights won over time to show emergence of hierarchy
fig1,ax1 = plt.subplots()
for f in range(len(fishes)):
    win_record = np.array(tank.fishes[f].win_record)[:,1]
    c_kernel = np.array([1,1,1,1,1,1,1,1,1])
    smooth_record = np.convolve(win_record,c_kernel,mode='valid') / sum(c_kernel)
    ax1.plot(smooth_record)
    ax1.set_xlabel('time (n fights)')
    ax1.set_ylabel('Proportion of fights won (sliding bin)')
## Figure 3b: Effort over time, to show that bayes becomes efficient
fig2,ax2 = plt.subplots()
effort_array = np.zeros([len(fishes),len(win_record)])
cost_array = np.zeros([len(fishes),len(win_record)])
for f in range(len(fishes)):
    cost_record = np.array(tank.fishes[f].win_record)[:,3]
    effort_record = np.array(tank.fishes[f].win_record)[:,2]
    effort_array[f] = effort_record
    cost_array[f] = cost_record
    ax2.plot(cost_record,color=cm.tab10(f),alpha=.2)
    #ax2.plot(effort_record,color=cm.tab10(f),alpha=0.4,linestyle=':')
    ax2.set_xlabel('time (n fights)')
    ax2.set_ylabel('Cost (actual energy spent)')
from scipy.ndimage import gaussian_filter1d

smooth_costs = gaussian_filter1d(np.median(cost_array,0),3)
ax2.plot(smooth_costs,color='black')

## Figure 3c: Estimate over time, to show that bayes is accurate

fig3,ax3 = tank.plot_estimates(food=False)
fig4,ax4 = plt.subplots()
ax4.plot(fishes[f].xs,fishes[f].prior)
ax4.axvline(fishes[f].range_record[-1][1])
ax4.axvline(fishes[f].range_record[-1][2])

if PLOT:
    plt.show()
