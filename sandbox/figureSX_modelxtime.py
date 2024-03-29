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
params.outcome_params = [0,0,-0.99]
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
params.n_fights = 200
## Set up a tank

if False:
    params.prior = True
    #fishes = [Fish(f,age=50,size=47,prior=True,insight=True,acuity=0,effort_method=params.effort_method,fight_params=params.outcome_params) for f in range(5)]
    fishes = [Fish(f,params) for f in range(5)]
   
else:
    #fishes = [Fish(f,insight=True,acuity=0,effort_method=params.effort_method,fight_params=params.outcome_params) for f in range(5)]
    fishes = [Fish(f,params) for f in range(5)]

#tank = Tank(fishes,n_fights = 50,death=False,f_method=params.f_method,f_params=params.outcome_params)
#import pdb;pdb.set_trace()
tank = Tank(fishes,params)
tank.run_all(print_me=False)

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
    ax2.plot(cost_record+f,color=cm.tab10(f),alpha=.2)
    #ax2.plot(effort_record,color=cm.tab10(f),alpha=0.4,linestyle=':')
    ax2.set_xlabel('time (n fights)')
    ax2.set_ylabel('Cost (actual energy spent)')
from scipy.ndimage import gaussian_filter1d

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

print(np.shape(tank.win_record))
window = 2 
n_windows = len(tank.history) // window * 2
lin_list = []
for i in range(n_windows):
    idx = slice(int(i*window/2),int(i*window/2 + window))
    linearity,[d,p] = sim._calc_linearity(tank,idx)
    lin_list.append(linearity)

fig5,ax5 = plt.subplots()
ax5.plot(lin_list)

PLOT = True
if PLOT:
    plt.show()
