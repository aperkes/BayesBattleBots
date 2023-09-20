#! /usr/bin/env python

## Code for fig 6a, showing how size of opponent impacts the strength of winner effect on different individuals


import numpy as np
from matplotlib import pyplot as plt

from params import Params
from fish import Fish,FishNPC
from fight import Fight
import copy

from tqdm import tqdm
from joblib import Parallel, delayed

#sizes = np.linspace(1,100)
sizes = [40,60]

params = Params()
params.prior = True
params.size = 50

## Make lots of NPC fish of different size
npcs = [FishNPC() for s in sizes]
for n_ in range(len(npcs)):
    npcs[n_].size = sizes[n_]

iterations = 3
#strategies = ['bayes','linear','no update']

def mean_sem(a):
    mean_a = np.nanmean(a,axis=0)
    sem_a = np.nanmean(a,axis=0) / np.sqrt(len(a))
    return mean_a,sem_a

def run_sim(params):
    f_params = copy.deepcopy(params)
    outcome_array = np.empty([iterations,2,2]) ## iterations x w/l x sizes
    outcome_array.fill(np.nan)
    platos_fish = Fish(0,f_params)
    assay_opp = Fish(1,f_params) 
    for o_ in range(len(sizes)):
        opp = npcs[o_]
        for i in range(iterations):
## Stage a fight
            #focal_fish = Fish(0,f_params)
            #focal_loser = Fish(1,f_params)
            focal_fish = copy.deepcopy(platos_fish)
            focal_loser = copy.deepcopy(platos_fish)
            stage_fight = Fight(opp,focal_fish,f_params,outcome=1)
            stage_fight.run_outcome()
            focal_fish.update(True,stage_fight)

            stage_loss = Fight(opp,focal_loser,f_params,outcome=0)
            stage_loss.run_outcome()
            focal_loser.update(False,stage_loss)

## ASsay against naive size matched fish (with accuractely centered prior)
            assay_fight = Fight(assay_opp,focal_fish,f_params)
            outcome = assay_fight.run_outcome()
            outcome_array[i,1,o_] = outcome

            assay_loser = Fight(assay_opp,focal_loser,f_params)
            outcome_l = assay_loser.run_outcome()
            outcome_array[i,0,o_] = outcome_l
    return outcome_array

s_res = 11
l_res = s_res
a_res = s_res
c_res = a_res

s_list = np.linspace(0,1,s_res)
l_list = np.linspace(-1,1,l_res)
a_list = np.linspace(0,1,a_res)
c_list = np.linspace(0,1,c_res)

## The last two are winners/losers x mean/sem
all_results = np.empty([s_res,l_res,a_res,c_res,2,2])
def run_many_sims(s):
    many_results = np.empty([l_res,a_res,c_res,2,2])
    params = Params()
    for l_ in range(l_res):
        l = l_list[l_]
        for a_ in range(a_res):
            a = a_list[a_]
            a_list[a_]
            for c_ in range(c_res):
                c = c_list[c_]
                params.outcome_params = [s,0.5,l]
                params.awareness = a
                params.acuity = c    
                params.set_params()
                some_results = run_sim(params)
                some_diff = np.abs(some_results[:,:,0] - some_results[:,:,1])
                many_results[l_,a_,c_] = mean_sem(some_diff)
    return many_results

if True:
    for s_ in tqdm(range(s_res)):
        s = s_list[s_]
        all_results[s_] = run_many_sims(s)
else:
    all_results = Parallel(n_jobs=11)(delayed(run_many_sims)(s) for s in s_list)
    all_results = np.array(all_results)
#import pdb;pdb.set_trace()

mean_winners = all_results[:,:,:,:,1,0]
sem_winners = all_results[:,:,:,:,1,1]
mean_losers = all_results[:,:,:,:,0,0]
sem_losers = all_results[:,:,:,:,0,1]

fig,axes = plt.subplots(2,2,sharey = 'row',sharex = 'col')
vmin =0
vmax = 1
axes[0,0].imshow(mean_winners[:,:,5,1],vmin=vmin,vmax=vmax)
axes[0,1].imshow(mean_winners[7,1,:,:],vmin=vmin,vmax=vmax)
axes[1,0].imshow(mean_losers[:,:,5,1],vmin=vmin,vmax=vmax)
axes[1,1].imshow(mean_losers[7,1,:,:],vmin=vmin,vmax=vmax)

plt.show()

## Check against a fair fish
## Run them against a naive, size-matched fish to test for winner effect  
# (do I do win percent, or just contrast winner vs loser)
