#! /usr/bin/env python

## Code for fig 6a, showing how size of opponent impacts the strength of winner effect on different individuals


import numpy as np
from matplotlib import pyplot as plt

from params import Params
from fish import Fish,FishNPC
from fight import Fight
from tqdm import tqdm

sizes = np.linspace(1,100)

params = Params()
params.prior = True
params.size = 50

## Make lots of NPC fish of different size
npcs = [FishNPC() for s in sizes]
for n_ in range(len(npcs)):
    npcs[n_].size = sizes[n_]

iterations = 1000
strategies = ['bayes','linear','no update']
## Make lots of focal fish with different strategies

outcome_array = np.empty([len(sizes),len(strategies),iterations])
outcome_array.fill(np.nan)

for o_ in tqdm(range(len(sizes))):
    opp = npcs[o_]
    for s_ in range(len(strategies)):
        f_params = params.copy()
        f_params.update_method = strategies[s_]
        for i in range(iterations):
## Stage a fight
            focal_fish = Fish(0,f_params)
            stage_fight = Fight(opp,focal_fish,f_params,outcome=1)
            stage_fight.run_outcome()
            focal_fish.update(True,stage_fight)

## ASsay against naive size matched fish (with accuractely centered prior)
            assay_opp = Fish(1,f_params) 
            assay_fight = Fight(assay_opp,focal_fish,f_params)
            outcome = assay_fight.run_outcome()
            outcome_array[o_,s_,i] = outcome

fig,ax = plt.subplots()

for s_ in range(len(strategies)):
    mean_outcome = np.mean(outcome_array[:,s_],axis=1)
    sem_outcome = np.std(outcome_array[:,s_],axis=1) / np.sqrt(iterations)
    ax.plot(sizes,mean_outcome,label=strategies[s_])            
    ax.fill_between(sizes,mean_outcome - sem_outcome,mean_outcome+sem_outcome,alpha=0.5,color='grey')

ax.set_xlabel('Size of opponent for staged win')
ax.set_ylabel('Probability of winning vs. size-matched opponent')
ax.legend()

fig.savefig('./figures/fig6a_discrepency.png',dpi=300)
plt.show()
## Check against a fair fish
## Run them against a naive, size-matched fish to test for winner effect  
# (do I do win percent, or just contrast winner vs loser)
