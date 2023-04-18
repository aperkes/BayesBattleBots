#! /usr/bin/env python

## Code for fig 6a, showing how size of opponent impacts the strength of winner effect on different individuals


import numpy as np
from matplotlib import pyplot as plt

from params import Params
from fish import Fish,FishNPC
from fight import Fight
from tqdm import tqdm

n_fights = np.arange(0,50,5)

params = Params()
opp_params = Params()

#params.prior = True
#params.size = 50
params.poly_param_m = 0.2
## Make lots of naive fish of random sizes
npcs = [Fish(s,opp_params) for s in range(int(max(n_fights) * 2))]

staged_opp = FishNPC(0,params)
staged_opp.size = 51
iterations = 1000
strategies = ['bayes','linear','no update']
## Make lots of focal fish with different strategies

outcome_array = np.empty([len(n_fights),len(strategies),iterations])
outcome_array.fill(np.nan)

for n_ in tqdm(range(len(n_fights))):
    for s_ in range(len(strategies)):
        f_params = params.copy()
        f_params.update_method = strategies[s_]
        for i in range(iterations):
            opps = np.random.choice(npcs,n_fights[n_])
            focal_fish = Fish(0,f_params)
## Provide background experiecne
            for o in opps:
                pre_fight = Fight(o,focal_fish,f_params)
                outcome = pre_fight.run_outcome()
                focal_fish.update(outcome,pre_fight)
## Run a staged fight
            pre_est = focal_fish.estimate
            staged_opp.size = focal_fish.size
            staged_opp.params.size = focal_fish.size
            staged_fight = Fight(staged_opp,focal_fish,outcome=1)
            staged_fight.run_outcome()
            focal_fish.update(True,staged_fight)
            #print(n_,s_,i,'pre,post:',pre_est,focal_fish.estimate)
## ASsay against naive size matched fish (with accuractely centered prior)
            assay_opp = Fish(1,f_params) 
            assay_fight = Fight(assay_opp,focal_fish,f_params)
            outcome = assay_fight.run_outcome()
            outcome_array[n_,s_,i] = outcome

fig,ax = plt.subplots()

for s_ in range(len(strategies)):
    mean_outcome = np.mean(outcome_array[:,s_],axis=1)
    sem_outcome = np.std(outcome_array[:,s_],axis=1) / np.sqrt(iterations)
    ax.plot(n_fights,mean_outcome,label=strategies[s_])            
    ax.fill_between(n_fights,mean_outcome - sem_outcome,mean_outcome+sem_outcome,alpha=0.5,color='grey')

ax.set_xlabel('Number of fights prior to assay')
ax.set_ylabel('Probability of winning vs. size-matched opponent')
ax.legend()

fig.savefig('./figures/fig6b_experience.png',dpi=300)
plt.show()
## Check against a fair fish
## Run them against a naive, size-matched fish to test for winner effect  
# (do I do win percent, or just contrast winner vs loser)
