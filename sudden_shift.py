#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from tqdm import tqdm

import random, copy

s,e,l = .6,.3,.01

params = SimParams()
params.effort_method = [1,1]
params.n_fights = 50 
params.n_iterations = 15
params.n_fish = 5
params.f_method = 'balanced'
params.u_method = 'bayes'
params.f_outcome = 'math'
params.outcome_params = [s,e,l]
s = Simulation(params)

winners,losers = [],[]

## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
iterations = 100
scale = 2
#for i in tqdm(range(iterations)):
if True:
    fishes = [Fish(f,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]
    tank = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    tank.run_all(False)

    for f in tank.fishes:
        f.old_size = f.size
        f.size = f.size + f.size * (random.random() - .5)

    tank2 = Tank(fishes,n_fights = params.n_fights * 2,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    tank2.run_all(False)

fig,ax = plt.subplots()
#for f in fishes:
n_rounds = params.n_fights * (params.n_fish-1)+1
xs = np.arange(n_rounds-1,n_rounds*2)
for i in range (len(fishes)):
    f = fishes[i]
    ax.plot(np.array(f.win_record)[:,2],color=cm.tab10(i))
    ax.plot([0,n_rounds],[f.old_size/100,f.old_size/100],color=cm.tab10(i))
    ax.plot([n_rounds,len(f.est_record)],[f.size/100,f.size/100],color=cm.tab10(i))
    #ax.fill_between(np.arange(len(f.est_record)),np.array(f.est_record_) + np.array(f.sdest_record),
    #                                             np.array(f.est_record_) - np.array(f.sdest_record),color=cm.tab10(i),alpha=.3)

#print(np.array(f.win_record)[:,2])
ax.axvline(params.n_fights * (params.n_fish-1),color='red',label='disruption')

ax.legend()
fig.savefig('test_disruption_fixed.jpg',dpi=300)
fig.show()
plt.show()
