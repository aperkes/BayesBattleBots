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
- Start with a Naive Strategy
- Seed with mutant, potentially better strategy
- Run tank, use the fitness to determine the ratio of the next tank
- Run 100 generations or until fixation
- Repeat 100x to get fixation probability
- Increase the cost of the mutant strategy until it can't compete
- Output 
    a) Whether it can invade
    b) the cost at which it can no longer invade 
"""

params = Params()

params.n_fights = 10 
params.n_fish = 20
params.f_method = 'shuffled'
params.xs = np.linspace(7,100,100)

## Estimate invasion
if True:
    params.effort_method = [None,0.1]
    params.update_method = None
    params.mutant_update = None

params.f_outcome = 'math'
params.outcome_params = [0.6,0.3,.05]
params.generations = 100
params.iterations = 10 
params.fitness_ratio = 0.1
params.death=True
params.food=1
params.free_food = 0.0
#params.baseline_effort = params.free_food
params.baseline_effort = 0.1
#params.baseline_weight,params.assessment_weight = [.9,.1]
params.assessment_weight = 0.9

## Parameters to mutate, not sure how they're used yet.
params.poly_param_a = 0.5
params.poly_param_b = 0.5
mutation_cost = .1

fig,ax = plt.subplots()

for mutation_cost in [0.0]:
    ess_count = 0
    gen_count = []
    m_trajectories = np.empty([params.iterations,params.generations])
    m_trajectories.fill(np.nan)
#for i in range(params.iterations):
    pilot_fish = Fish(0,params)
    params.naive_likelihood = pilot_fish.naive_likelihood
    efforts = np.linspace(0,1,params.n_fish)
    for i in tqdm(range(params.iterations)):
        count = 0
        fishes = [Fish(f,params) for f in range(params.n_fish)]
        for f_ in range(len(fishes)):
            f = fishes[f_]
            f.params.baseline_effort = efforts[f_]
            f.params.effort_method = [None,f.params.baseline_effort]
        while count < params.generations - 1:
            tank = Tank(fishes,params)
            tank._initialize_likelihood()

            tank.run_all(progress=False)

            #print(fishes[0].fitness_record)
            fitness = np.array([sum(f.fitness_record) for f in fishes])
            rel_fitness = fitness / np.sum(fitness)
            n_new_fish = (rel_fitness * params.n_fish).astype(int)
            alive = [f.alive for f in fishes]
            

            #mutant_fitness = sum([fitness[m] for m in range(n_mutants)])
            #other_fitness = sum(fitness[1:])
            #mutant_ratio = mutant_fitness / sum(fitness)
            #n_mutants = int(params.n_fish * mutant_ratio)

            #print('mutant fitness',mutant_fitness)
            #print('total fitness',sum(fitness),fitness)
            #print('n_mutant offspring:',n_mutants, 'of',params.n_fish)
            count += 1
            #print(n_mutants)
            new_fishes = []
            possible_fish = []
            for f_ in range(len(fishes)):
                possible_fish.extend([f_]*fitness[f_])
            new_fish_idx = random.sample(possible_fish,params.n_fish)
            new_fish = [Fish(f_,fishes[f_].params) for f_ in new_fish_idx]
            fishes = new_fish


    print('final count:',ess_count,'of',params.iterations)
    m_trajectories = m_trajectories / params.n_fish
    y_mean = np.nanmean(m_trajectories,0)
    y_err = np.nanstd(m_trajectories,0) / np.sum(~np.isnan(m_trajectories),0)
    ax.plot(y_mean,color=cm.viridis(mutation_cost),label=str(mutation_cost))
    ax.fill_between(np.arange(params.generations),y_mean+y_err,y_mean-y_err,color=cm.viridis(mutation_cost),alpha=.5)
    for i in range(params.iterations):
        ax.plot(m_trajectories[i],color=cm.viridis(mutation_cost),alpha=.2)

ax.set_ylim([-0.1,1.1])
fig.legend()
fig.show()
plt.show()


