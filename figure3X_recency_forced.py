#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation
from params import Params

import numpy as np
from scipy.stats import binom_test

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy
import itertools

#s,e,l = .6,.3,.1

#s,e,l = -0.5,0.5,-0.9
## I will need to iterate across all possible params. Yikes. 

params = Params()

#params.boldness = 0.7
params.effort_method = 'LuckyPoly' ## This gets the average in one line without them having 100% conf. 

params.energy_cost = False
params.prior=True ## Their estimate also starts accurate

params.iterations = 2
params.n_rounds = 6

params.f_outcome = 'math'

winners,losers = [],[]

## Have a fish win-lose, or lose-win, then measure the winner effect for that fish

params.age = 51
params.size = 50
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

s_set = [-1,-0.3,0,0.3,1]
e_set = list(s_set)
l_set = list(s_set)
a_set = [0,0.5,1]
c_set = list(a_set)
n_C = len(s_set) ## as in density of dimensionality
n_A = len(a_set)

decay_array = np.empty([n_C,n_C,n_C,n_A,n_A])
wl_array = np.empty_like(decay_array)
rec_array = np.empty_like(decay_array)
late_array = np.empty_like(decay_array)

c_indices = range(n_C)
a_indices = range(n_A)
base_results = build_results(params.n_rounds)
n_results = [[] for m in range(params.n_rounds)]
for k in base_results.keys():
    for m in range(params.n_rounds):
        if len(k) == m:
            n_results[m].append(k)

for (s_,e_,l_,a_,c_) in itertools.product(c_indices,c_indices,c_indices,a_indices,a_indices):
    s,e,l = s_set[s_],e_set[e_],l_set[l_]
    a,c = a_set[a_],c_set[c_]
    print(s,e,l,a,c)

    params.awareness = a
    params.acuity = c
    params.outcome_params = [s,e,l]
    params.set_params()

    f0 = Fish(1,params)
    fishes = {'':build_fish(1,f0)} 
    previous_round = ['']
    match_results = dict(base_results)
    match_prob = {}
## NOTE: I only actually need as many iterations as 
# there are permutations, if I were smarter about this
#for i in tqdm(range(iterations)):
    results_str = ''
    f = build_fish(1,f0)
    f_match = build_fish(0,f0)

    ## Run all matches
    for m in range(1,params.n_rounds):
        this_round = n_results[m]
        for exp in previous_round:
            f = fishes[exp]
            for w in [0,1]: ## force one of two conditions
                f_ = build_fish(1,f)
                match,_ = check_success(f_,f_match,params)
                results_str = exp + conversion_dict[w]
                f_.update(1-w,match) 
                fishes[results_str] = f_
            if match.f_min:
                p_win = 1 - match.p_win
            else:
                p_win = match.p_win
            match_prob[exp] = p_win
            previous_round = this_round
    wl_effect = match_prob['w'] - match_prob['l']
    rec_effect = match_prob['lw'] - match_prob['wl']
    rec_effect_late = match_prob['llw'] - match_prob['wwl']
    rec_decay = np.abs(rec_effect_late) * np.sign(rec_effect) * np.sign(rec_effect_late)

## This happens when a certain event never occurs
    #if np.isnan(rec_decay):
    #    import pdb;pdb.set_trace()
    decay_array[s_,e_,l_,a_,c_] = rec_decay
    wl_array[s_,e_,l_,a_,c_] = wl_effect
    rec_array[s_,e_,l_,a_,c_] = rec_effect
    late_array[s_,e_,l_,a_,c_] = rec_effect_late

import pdb;pdb.set_trace()
