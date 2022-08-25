#! /usr/bin/env python

""" Code to run ESS simulations 
written by Ammon Perkes
contact perkes.ammon@gmail.com for questions """

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

import numpy as np

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

params = SimParams()

params.n_fights = 5
params.n_fish = 20
params.f_method = 'shuffled'

## Bayes invasion
if True:
    params.effort_method = 'Estimate'
    params.acuity = 5
    params.awareness = 20
    params.mutant_effort = 'BayesMA'

    params.u_method = None
    params.mutant_update = 'bayes'

## Estimate invasion
elif True:
    params.effort_method = [None,0.5]
    #params.mutant_effort = 'Perfect' ## Obviously perfect is better. 
    params.mutant_effort = 'Estimate'
    params.acuity = 20
    params.awareness = 20

    params.u_method = None
    params.mutant_update = None
## Bayes invation
elif True:
    params.effort_method = [None,0.5]
    params.mutant_effort = [1,0]

    params.u_method = None
    params.mutant_update = 'bayes'
else: ## .5 invasion of 100%
    params.effort_method = [None,None]
    params.mutant_effort = [None,0.5]

    params.u_method = None
    params.mutant_update = None

params.f_outcome = 'math'
params.outcome_params = [0.6,0.3,.05]
params.generations = 50
params.iterations = 10 


mutation_cost = .1

fig,ax = plt.subplots()

for mutation_cost in [0.0,0.2]:
    ess_count = 0
    gen_count = []
    m_trajectories = np.empty([params.iterations,params.generations])
    m_trajectories.fill(np.nan)
#for i in range(params.iterations):
    for i in tqdm(range(params.iterations)):
        count = 0
        n_mutants = 15
        m_trajectories[i,0] = n_mutants
        while n_mutants < params.n_fish and count < params.generations - 1:
            if n_mutants == 0:
                break
            pilot_fish = Fish(0,effort_method=params.effort_method,fight_params=params.outcome_params)
            fishes = [Fish(f,prior=params.awareness,likelihood = pilot_fish.naive_likelihood,fight_params=params.outcome_params,effort_method=params.effort_method,update_method=params.u_method,acuity=params.acuity,awareness=params.awareness) for f in range(params.n_fish)]
            for m in range(n_mutants):
                fishes[m] = Fish(m,prior=20,likelihood = pilot_fish.naive_likelihood,fight_params=params.outcome_params,effort_method=params.mutant_effort,update_method=params.mutant_update,max_energy=1-mutation_cost,c_aversion=1,acuity=params.acuity,awareness=params.awareness)

            tank = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_outcome=params.f_outcome,f_method=params.f_method,u_method=params.u_method,fitness_ratio=0.1,death=True,food=0.1)
            tank._initialize_likelihood()

            tank.run_all(progress=False)

            fitness = [sum(f.fitness_record) for f in fishes]
            alive = [f.alive for f in fishes]

            mutant_fitness = sum([fitness[m] for m in range(n_mutants)])
            print('mutant fitness',mutant_fitness)
            other_fitness = sum(fitness[1:])
            mutant_ratio = mutant_fitness / sum(fitness)
            n_mutants = int(params.n_fish * mutant_ratio)
            print('total fitness',sum(fitness),fitness)
            print('n_mutant offspring:',n_mutants, 'of',params.n_fish)
            count += 1
            #print(n_mutants)
            m_trajectories[i,count] = n_mutants
        m_trajectories[i,count:] = n_mutants 
        if n_mutants == params.n_fish:
            #print('ESS Acheived!')
            ess_count += 1
            gen_count.append(count)

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


