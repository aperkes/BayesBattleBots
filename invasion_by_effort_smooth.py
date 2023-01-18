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

#######
""" Design:
- Start with a Fixed effort of some level
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
## Bayes invasion
## Estimate invasion
if True:
    params.effort_method = [None,0.1]
    params.mutant_effort = 'SmoothPoly' ## Obviously perfect should be better. 
    #params.mutant_effort = 'EstimatePoly'
    params.acuity = 5
    params.awareness = 5

    params.update_method = None
    params.mutant_update = None

params.f_outcome = 'math'
params.outcome_params = [0.6,0.3,.05]
params.generations = 50
params.iterations = 10 
params.fitness_ratio = 0.1
params.death=True
params.food=1
params.free_food = 0.0
#params.baseline_effort = params.free_food
params.baseline_effort = 0.1
#params.baseline_weight,params.assessment_weight = [.9,.1]
params.assessment_weight = 0.9

mutation_cost = .1

pilot_fish = Fish(0,params)
params.naive_likelihood = pilot_fish.naive_likelihood
#for baseline_effort in [0.01]:
#for baseline_effort in [0,0.01,0.2,0.4,0.6,0.8,1.0]:
for baseline_effort in [0,0.1,0.8,1.0]:
    print('Baseline effort:',baseline_effort)
    fig,ax = plt.subplots()
    #for mutation_cost in [0.0,0.1,0.2,0.3]:
    for mutation_cost in [0.0]:
        print('Effort:',baseline_effort,'Cost:',mutation_cost)
        params.baseline_effort = baseline_effort
        ess_count = 0
        gen_count = []
        m_trajectories = np.empty([params.iterations,params.generations])
        m_trajectories.fill(np.nan)
#for i in range(params.iterations):
        mutant_params = params.copy()
        mutant_params._mutate()
        mutant_params.max_energy = mutant_params.max_energy - mutation_cost
        for i in tqdm(range(params.iterations)):
            count = 0
            n_mutants = 2
            m_trajectories[i,0] = n_mutants
            while n_mutants < params.n_fish and count < params.generations - 1:
                if n_mutants == 0:
                    break
                fishes = [0 for i in range(params.n_fish)]
                for m in range(n_mutants):
                    fishes[m] = Fish(m,mutant_params)
                for n in range(n_mutants,params.n_fish):
                    fishes[n] = Fish(n,params)
                if baseline_effort == 0:
                    for n in range(n_mutants,params.n_fish):
                        fishes[n].params.baseline_effort = np.random.random()

                tank = Tank(fishes,params)
                tank._initialize_likelihood()

                tank.run_all(progress=False,print_me=False)

                #print(fishes[0].fitness_record)
                fitness = [sum(f.fitness_record) for f in fishes]
                alive = [f.alive for f in fishes]

                print(fitness)
                mutant_fitness = sum([fitness[m] for m in range(n_mutants)])
                other_fitness = sum(fitness[1:])
                mutant_ratio = mutant_fitness / sum(fitness)
                n_mutants = int(params.n_fish * mutant_ratio)
                print('n mutants:',n_mutants)

                #print('mutant fitness',mutant_fitness)
                #print('total fitness',sum(fitness),fitness)
                #print('n_mutant offspring:',n_mutants, 'of',params.n_fish)
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
    fig.savefig('FigEstimateEss_'+str(baseline_effort)+'.png',dpi=300)
plt.show()


