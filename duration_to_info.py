#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import copy

s,e,l = .6,.3,.01

params = SimParams()
params.effort_method = [1,1]
params.n_fights = 5 
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
scale = 1
for i in tqdm(range(iterations)):
    fishes = [Fish(f,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]
    tank = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    tank.run_all(False)

#for f in tank.fishes:
    f = tank.fishes[0]
    f2 = tank.fishes[1]
    opp = Fish(size = f.size*scale)
    opp2 = Fish(size = f2.size/scale)
    f_win = Fight(f,opp,outcome=0)
    f_loss = Fight(f,opp2,outcome=1) 
    f_win.winner = f
    f_win.loser = opp
    f_loss.winner = opp2
    f_loss.loser = f2
    #print(f.estimate,len(f.est_record))
    f.update(True,f_win)    
    f2.update(False,f_loss)
    #print(f.estimate)
    tank2 = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    tank3 = copy.deepcopy(tank2)
    tank2.run_all(False)
    tank3.run_all(False)
    winners.append(f)
    losers.append(f2)
fig,ax = plt.subplots()
#for f in fishes:
n_rounds = params.n_fights * (params.n_fish-1)+1
xs = np.arange(n_rounds-1,n_rounds*2)
for f in winners:
    #f = tank.fishes[0]
    ax.plot(np.array(f.est_record)-f.est_record[n_rounds-1],color='green',alpha=.1)
for f in losers:
    ys = np.array(f.est_record) - f.est_record[n_rounds-1]
    ax.plot(ys,color='purple',alpha=.1)
ax.axvline(params.n_fights * (params.n_fish-1),color='red',label='forced win')
ax.axhline(0,color='black',label='estimate prior to staged fight')
ax.set_xlim([n_rounds-5,n_rounds+5])
ax.set_ylim([-7.1,7.1])
ax.legend()
fig.savefig('test_staged.jpg',dpi=300)
fig.show()
