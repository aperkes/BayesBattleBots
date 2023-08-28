#! /usr/bin/env python

## Code for fig 3b, showing the loser effect is stronger

import numpy as np
from matplotlib import pyplot as plt

from fish import Fish,FishNPC
from fight import Fight
from params import Params

from tqdm import tqdm

iterations = 1000
params = Params()
params.outcome_params = [-0.5,0,-0.4]
params.set_params()
print(params.outcome_params)
#params.size = 50
#assay_fish = Fish(1,params)

assay_params = params.copy()

outcome_array = np.empty([iterations,2])
outcome_array.fill(np.nan)

win_info_array = np.array(outcome_array)
loss_info_array = np.array(outcome_array)
iterations = 1000
for i in tqdm(range(iterations)):
    focal_winner = Fish(i+2,params)
    focal_loser = focal_winner.copy() 
## Stage a bunch of wins and losses against size-matched fish

    #assay_params.size = focal_winner.size
    staged_opp = FishNPC(0,assay_params)
    staged_opp.size = focal_winner.size

    staged_win = Fight(staged_opp,focal_winner,params,outcome=1)
    staged_win.run_outcome()
    focal_winner.update(True,staged_win)

    staged_loss = Fight(staged_opp,focal_loser,params,outcome=0)
    staged_loss.run_outcome()
    focal_loser.update(False,staged_loss)

## Assay against size matched fish
    assay_params.size = focal_winner.size
    assay_fish = Fish(1,assay_params)

    assay_winner = Fight(assay_fish,focal_winner,params)
    winner_output = assay_winner.run_outcome()
    outcome_array[i,1] = winner_output

    #print('win:',assay_fish.effort,focal_winner.effort,winner_output) 
    assay_loser = Fight(assay_fish,focal_loser,params)
    loser_output = assay_loser.run_outcome()
    outcome_array[i,0] = loser_output
    win_info_array[i] = focal_winner.effort,focal_winner.estimate 
    loss_info_array[i] = focal_loser.effort,focal_loser.estimate 
    #print('loss:',assay_fish.effort,focal_loser.effort,loser_output)
    #break
print(np.mean(outcome_array,axis=0))
print(np.std(outcome_array,axis=0))

means = np.mean(outcome_array,axis=0)-0.5
means_ = -1 * means
sems = np.std(outcome_array,axis=0) / np.sqrt(iterations)

fig,ax = plt.subplots()

ax.bar([0,1],means,yerr = sems,bottom = 0.5,color = ['darkblue','gold'])
ax.bar([0,1],means_,yerr = sems,bottom = 0.5,color = ['darkblue','gold'],alpha=0.1)
ax.axhline(0.5,color='black',linestyle=':')

ax.set_xticks([0,1])
ax.set_xticklabels(['Loser','Winner'])
ax.set_ylabel('Probability of win vs size-matched fish')

#fig.savefig('./figures/fig3b_winvloss.png',dpi=300)
print('win effort, win estimate')
print(np.mean(win_info_array,axis=0))
print(np.mean(loss_info_array,axis=0))
if True:
    plt.show()
