#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
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
PLOT = 'estimate'
NAIVE = True

## Let fish duke it out, then pull a fish out, check it's success against a size matched fish, and put it back in
iterations = 30
scale = 1
f0 = Fish(0,effort_method=params.effort_method,update_method=params.u_method)
results = [[],[]]
if NAIVE:
    pre_rounds = 2
else:
    pre_rounds = 50
for i in tqdm(range(iterations)):
#for i in range(iterations):
    fishes = [Fish(f,likelihood=f0.naive_likelihood,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]
    tank = Tank(fishes,n_fights = pre_rounds,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    tank._initialize_likelihood() ## This is super important, without intiializing it, it takes ages. 
    tank.run_all(False)

    focal_fish = fishes[0]
    i_shift = len(focal_fish.est_record) - 1
    matched_fish = Fish(0,size=focal_fish.size,prior=True,effort_method=params.effort_method,update_method=params.u_method)
    match = Fight(focal_fish,matched_fish,outcome_params=params.outcome_params,outcome=np.random.randint(2))
    outcome = match.run_outcome()
    focal_fish.update(1-outcome,match)
    match2 = Fight(focal_fish,matched_fish,outcome_params=params.outcome_params)
    test_outcome = match2.run_outcome()
    if outcome == 0:
        winners.append(focal_fish)
    else:
        losers.append(focal_fish)
    results[1-outcome].append(1-test_outcome)

    tank2 = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    tank2.run_all(False)


print('winner win-rate:',np.mean(results[1]))
print('loser win-rate:',np.mean(results[0]))

fig,ax = plt.subplots()

for f in losers:
    ax.plot(f.est_record - f.est_record[i_shift],color='darkblue',alpha=.4)
for f in winners:
    ax.plot(f.est_record - f.est_record[i_shift],color='gold',alpha=.4)

ax.axvline(i_shift)
ax.set_xlim([i_shift - 2,i_shift + 10])
ax.set_xticks(np.arange(i_shift -2,i_shift + 10 + 1,2))
ax.set_xticklabels([-2,0,2,4,6,8,10])
ax.set_xlabel('fights since size-matched challenge')
ax.set_ylabel('Difference in estimate')
plt.show()

