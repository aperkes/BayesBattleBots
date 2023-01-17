#! /usr/bin/env python

""" Code to run ESS simulations 
written by Ammon Perkes
contact perkes.ammon@gmail.com for questions """

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from params import Params
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import numpy as np
import random
#######
""" Design:
Figure out the best level of investment, given perfect information
Start with random parameters, vary them to find the best ones
"""

params = Params()

params.n_fights = 10 
params.n_fish = 20
params.f_method = 'shuffled'
params.xs = np.linspace(7,100,100)

## Estimate invasion
if True:
    params.effort_method = 'PerfectPoly'
    params.update_method = None
    params.mutant_update = None

effort_bins = 11
params.f_outcome = 'math'
params.outcome_params = [0.6,0.3,.05]
params.generations = 300
params.iterations = 5 
params.fitness_ratio = 0.1
params.death=True
params.food=1
params.free_food = 0.0
#params.baseline_effort = params.free_food
params.baseline_effort = 0.1
#params.baseline_weight,params.assessment_weight = [.9,.1]
params.assessment_weight = 0.9

## Parameters to mutate, not sure how they're used yet.
if False: ## These are the best evolved parameters
    params.poly_param_a = 0.6
    params.poly_param_b = 0.1
else:
    params.poly_param_a = 0.9
    params.poly_param_b = 0.9 

mutation_cost = .1


#all_effort_maps = np.empty([params.iterations,params.generations,effort_bins])
#effort_history = np.empty([params.iterations,params.generations,params.n_fish])
param_a_history = np.empty([params.iterations,params.generations,params.n_fish])
param_b_history = np.empty([params.iterations,params.generations,params.n_fish])
for mutation_cost in [0.0]:
    pilot_fish = Fish(0,params)
    params.naive_likelihood = pilot_fish.naive_likelihood
    for i in tqdm(range(params.iterations)):
        count = 0
        fishes = [Fish(f,params) for f in range(params.n_fish)]
## optionally, start with random values
        for f_ in range(len(fishes)):
            f = fishes[f_]
            #f.params.poly_param_a = np.random.random()
            #f.params.poly_param_b = np.random.random()
            param_a_history[i,0,f_] = f.params.poly_param_a
            param_b_history[i,0,f_] = f.params.poly_param_b
        while count < params.generations - 1:
            tank = Tank(fishes,params)
            tank._initialize_likelihood()

            tank.run_all(progress=False)

            #print(fishes[0].fitness_record)
            fitness = np.array([sum(f.fitness_record) for f in fishes])
            fitness = fitness * 10
            #rel_fitness = fitness / np.sum(fitness)
            #n_new_fish = (rel_fitness * params.n_fish).astype(int)
            #alive = [f.alive for f in fishes]
            count += 1

            #print(fitness)
            possible_fish = []
            for f_ in range(len(fishes)):
                possible_fish.extend([f_]*fitness[f_])
            new_fish_idx = random.sample(possible_fish,params.n_fish)
            new_fish = [Fish(f_,fishes[f_].params) for f_ in new_fish_idx]
            for f in new_fish:
                f.mutate()
            fishes = new_fish
            for f_ in range(len(fishes)):
                f = fishes[f_]
                param_a_history[i,count,f_] = f.params.poly_param_a
                param_b_history[i,count,f_] = f.params.poly_param_b

print('mean last generation param a:',np.mean(param_a_history[:,-1,:]))
print('mean last generation param b:',np.mean(param_b_history[:,-1,:]))
print('std last generation param a:',np.std(param_a_history[:,-1,:]))
#print('std last generation:',np.std(effort_history[:,-1,:]))
#mean_map = np.transpose(np.nanmean(all_effort_maps,0))
#mean_map = np.flipud(mean_map)

print(np.histogram(param_a_history[:,0],bins=effort_bins))
heatmap_a = np.empty([params.generations,effort_bins])
heatmap_b = np.empty([params.generations,effort_bins])
for g in range(params.generations):
    hist_a,edges = np.histogram(param_a_history[:,g],range=[0,1],bins=effort_bins)
    heatmap_a[g] = hist_a
    hist_b,edges = np.histogram(param_b_history[:,g],range=[0,1],bins=effort_bins)
    heatmap_b[g] = hist_b

print(heatmap_b[-1])

if True:
    fig,(ax,ax1) = plt.subplots(2)
    ax.imshow(np.flipud(np.transpose(heatmap_a)))
    ax1.imshow(np.flipud(np.transpose(heatmap_b)))
#ax.set_ylim([-0.1,1.1])
#fig.legend()
    fig.savefig('strategy_evolution.png',dpi=300)
    #fig.show()
    #plt.show()
print('all done!')

