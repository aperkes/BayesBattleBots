#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from scipy.stats import binom_test

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy
import itertools

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

## Have a fish win-lose, or lose-win, then measure the winner effect for that fish

## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
iterations = 1000
scale = 2

upset_count = 0
win_count1,win_count2 = 0,0
win_shifts,loss_shifts = [],[]
## Lets assume their estimate is correct.

age = 50
size = 47
## Some helpful functions:
## Build copies of f0 with naive priors
def build_fish(idx,f0):
    return Fish(idx,size=f0.size,prior=True,effort_method=params.effort_method,fight_params=params.outcome_params,update_method=params.u_method,likelihood=f0.naive_likelihood)

def check_success(f,f_match):
    fight = Fight(f,f_match,outcome_params=params.outcome_params)
    fight.run_outcome()
    return fight,fight.outcome

## Tidy function to build a dict for storing everything
def build_results(n_matches=1):
    match_results = {}
    for d in range(n_matches):
        combos = [''.join(r) for r in itertools.product(['w','l'],repeat=d)]
        for c in combos:
            match_results[c] = []
    return match_results

## Dict to convert from outcome to letters
conversion_dict = {0:'w',1:'l'}

f0 = Fish(1,age=age,size=47,prior=True,effort_method=params.effort_method,fight_params=params.outcome_params,update_method=params.u_method)
n_matches = 1
fishes = []
match_results = build_results(n_matches)
for i in range(iterations):
#for i in tqdm(range(iterations)):
    results_str = ''
    f = build_fish(1,f0)
    f_match = build_fish(0,f0)

    ## Run all matches
    for m in range(n_matches):
        match,outcome =check_success(f,f_match)
        f.update(1-outcome,match)
        if f.est_record_[1] < 10:
            print('ITERATION ',i)
            print('MATCH ',m)
            print(f.est_record_)
            print(f.effort)
        #f.decay_prior(store=False)
        match_results[results_str].append(1-outcome)
        results_str += conversion_dict[outcome]

    fishes.append(f)
#print('round0 results:',match_results[''])
print('round one average:',np.mean(match_results['']))

fig,ax = plt.subplots()

loser_estimates, winner_estimates = [],[]

for f in fishes:
    jitter = np.random.randn()*.01
    jitter = 0
    ax.plot(np.array(f.est_record_)[:] + jitter,alpha=.01,color='black')
    if False:
        if f.win_record[0][1] == 1: ## If you won the first fight
            if f.win_record[1][1] == 0: # but lost the second fight
                winner_estimates.append(f.est_record_[2])
        elif f.win_record[1][1] == 1: # vs if you lost the first fight and won the second fight
            loser_estimates.append(f.est_record_[2])
    if True: ## Get estimates after the first fight
        if f.win_record[0][1] == 1: ## if you won the first fight:
            winner_estimates.append(f.est_record_[1])
        else:
            loser_estimates.append(f.est_record_[1])
#print(np.mean(match_results['wl']),np.mean(match_results['lw']))
print(np.mean(winner_estimates),np.mean(loser_estimates))
print('winners:',max(winner_estimates),min(winner_estimates),np.std(winner_estimates))
print('losers:',max(loser_estimates),min(loser_estimates),np.std(loser_estimates))

[ax.axvline(f,color='black',linestyle=':') for f in [0,1,2,3]]

ax.set_xticks([0,1,2,3])
#ax.set_xlim([-0.1,3.1])
ax.set_ylabel('Estimate (mm)')
ax.set_xlabel('Repeated Contests')

fig.show()
plt.show()
