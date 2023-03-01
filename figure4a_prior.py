#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation
from params import Params

import numpy as np
from scipy.stats import binom_test,norm

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy
import itertools

#s,e,l = .6,.3,.1

params = Params()
if True:
    params.effort_method = 'SmoothPoly'

params.poly_param_a = 2
params.poly_param_c = 0.1
params.n_fights = 50 
params.size = 50
params.energy_cost = False
params.acuity = 1
params.awareness = 5
params.start_energy = 1

params.iterations = 10000
params.n_fish = 5
params.f_method = 'balanced'
params.u_method = 'bayes'
params.f_outcome = 'math'
params.set_L()



## Have a fish win-lose, or lose-win, then measure the winner effect for that fish

## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
scale = 2

upset_count = 0
win_count1,win_count2 = 0,0
win_shifts,loss_shifts = [],[]
## Lets assume their estimate is correct.

params.age = 52
params.size = 50 

## Set up the various fish conditions
#size_params = params.copy() ## Fish determine prior from their size
self_params = params.copy() ## fish determine prior from their size/age

big_params = params.copy() ## Fish determine prior from surround fish (controlled to be big)
big_prior = norm.pdf(params.xs,75,5)
big_params.prior = big_prior / sum(big_prior)

small_params = params.copy() ## Fish determine prior from surrounding fish (controlled to be small)
small_prior = norm.pdf(params.xs,25,5)
small_params.prior = small_prior / sum(small_prior)

null_params = params.copy() ## Fish have null prior
guess_params = params.copy() ## Fish have random prior (but confident)

guess_params.awareness = 20
opp_params = params.copy()
opp_params.prior = True
opp_params.size = 35

## Uniform prior
null_params.prior = np.ones_like(null_params.xs) / len(null_params.xs)

self_fishes = [Fish(n+1,self_params) for n in range(params.iterations)]
null_fishes = [Fish(n+1,null_params) for n in range(params.iterations)]
big_fishes = [Fish(n+1,big_params) for n in range(params.iterations)]
small_fishes = [Fish(n+1,small_params) for n in range(params.iterations)]
guess_fishes = []
## Make sloppy fish confident
for n in tqdm(range(params.iterations)):
    f = Fish(n+1,guess_params)
    prior = f.prior ** 10
    f.prior = prior / np.sum(prior)
    f.get_stats()
    guess_fishes.append(f)

for n in tqdm(range(params.iterations)):
    f = Fish(n+1,guess_params)

opp_fish = Fish(0,opp_params)

#print(opp_fish.estimate)
#print(prior_fishes[0].estimate,null_fishes[0].estimate,guess_fishes[0].estimate)
p_efforts,n_efforts,g_efforts = [],[],[]
b_efforts,s_efforts = [],[]

for n in tqdm(range(params.iterations)):
    self_fight = Fight(self_fishes[n],opp_fish)
    self_fight.run_outcome()

    null_fight = Fight(null_fishes[n],opp_fish)
    null_fight.run_outcome()

    guess_fight = Fight(guess_fishes[n],opp_fish)
    guess_fight.run_outcome()

    big_fight = Fight(big_fishes[n],opp_fish)
    big_fight.run_outcome()

    small_fight = Fight(small_fishes[n],opp_fish)
    small_fight.run_outcome()
    p,o = self_fishes[n],null_fishes[n]
    g = guess_fishes[n]
    b,s = big_fishes[n],small_fishes[n]

    p_efforts.append(p.effort)
    n_efforts.append(o.effort)
    g_efforts.append(g.effort)
    b_efforts.append(b.effort)
    s_efforts.append(s.effort)

p_mean,p_std = np.mean(p_efforts),np.std(p_efforts)
n_mean,n_std = np.mean(n_efforts),np.std(n_efforts)
g_mean,g_std = np.mean(g_efforts),np.std(g_efforts)
print(np.mean(p_efforts),np.mean(n_efforts),np.mean(g_efforts))
print(np.std(p_efforts),np.std(n_efforts),np.std(g_efforts))

#fig,(ax,ax2,ax3) = plt.subplots(3)
fig,ax = plt.subplots()
#ax.bar([1,2,3],[p_mean,n_mean,g_mean],yerr=[p_std,n_std,g_std],alpha=.1)
ax.boxplot([n_efforts,g_efforts,p_efforts,s_efforts,b_efforts])
ax.set_xticklabels(['Uniform\nPrior','Random\nPrior','Self-Informed\nPrior','Small-Informed\nPrior','Big-Informed\nPrior'],rotation=45)
ax.axvline(2.5,linestyle=':',color='black')
#ax2.scatter([f.guess for f in prior_fishes],[f.effort for f in prior_fishes])
#ax3.scatter([f.estimate for f in prior_fishes],[f.effort for f in prior_fishes])
ax.set_ylabel('Effort')
fig.tight_layout()
fig.show()
plt.show()
"""
## Some helpful functions:
## Build copies of f0 with naive priors
def build_fish(idx,f0):
    return Fish(idx,f0.params)

def check_success(f,f_match,params):
    fight = Fight(f,f_match,params)
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

f0 = Fish(1,params)
n_matches = 3
fishes = []
match_results = build_results(n_matches)
for i in range(params.iterations):
#for i in tqdm(range(iterations)):
    results_str = ''
    f = build_fish(1,f0)
    f_match = build_fish(0,f0)

    ## Run all matches
    for m in range(n_matches):
        #print('f_before:',f.estimate)
        match,outcome =check_success(f,f_match,params)
        f.update(1-outcome,match)
        #f.decay_prior(store=False)
        #print(f.effort,f_match.effort)
        #print('f_after:',f.estimate)
        match_results[results_str].append(1-outcome)
        results_str += conversion_dict[outcome]

    fishes.append(f)
#print('round0 results:',match_results[''])
print('round one average:',np.mean(match_results['']))

fig,ax = plt.subplots()

loser_estimates, winner_estimates = [],[]

for f in fishes:
    jitter = np.random.randn() * 1
    jitter = 0
    ax.plot(np.array(f.est_record_)[:] + jitter,alpha=.008,color='black')
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

fig.savefig('./figures/Fig4b.png',dpi=300)
if False:
    fig.show()
    plt.show()
"""
