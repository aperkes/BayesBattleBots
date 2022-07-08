#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from tqdm import tqdm

import copy

s,e,l = .6,.3,.01

params = SimParams()
params.effort_method = [1,1]
params.n_fights = 5 
params.n_iterations = 15
params.n_fish = 5
params.f_method = 'balanced'
params.u_method = 'bayes'
params.f_outcome = 'math'
params.outcome_params = [s,e,l]
s = Simulation(params)

winners,losers = [],[]
biggers,smallers = [],[]
final_win,final_loss = [],[]

## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
iterations = 1000
scale = 1


plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'lines.linewidth':5})
plt.rcParams.update({'axes.linewidth':2})

fig,ax = plt.subplots()
fig.set_size_inches(17,11.5)

INIT = True

for i in tqdm(range(iterations)):
    fishes = [Fish(f,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]
    tank = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    if INIT:
        tank._initialize_likelihood()
    tank.run_all(False)

    f = tank.fishes[0]
    f2 = copy.deepcopy(tank.fishes[1])
    opp = Fish(size = f.size*scale)
    opp2 = Fish(size = f2.size/scale)
    f_win = Fight(f,opp,outcome=0)
    f_loss = Fight(f2,opp2,outcome=1) 
    f_win.winner = f
    f_win.loser = opp
    f_loss.winner = opp2
    f_loss.loser = f2
    #print(f.estimate,len(f.est_record))
    f.update(True,f_win)    
    f2.update(False,f_loss)
    #print(f.estimate)
    tank2 = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    if INIT:
        tank2._initialize_likelihood()
    fishes3 = copy.deepcopy(fishes)
    fishes3[1] = f2
    tank3 = Tank(fishes3,n_fights = params.n_fights,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
    if INIT:
        tank3._initialize_likelihood()
    #tank3 = copy.deepcopy(tank2)
    tank2.run_all(False)
    tank3.run_all(False)
    winners.append(f)
    rank_order= np.argsort(np.array(tank2.sizes))
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(tank2.n_fish) ## This took me an embarrasingly long time to get right. 
    f_rank = ranks[f.idx]
    l_rank = ranks[f2.idx]
    if np.max(tank2.sizes) != f.size:
        idx_bigger = np.arange(tank.n_fish)[ranks == (f_rank + 1)][0]
    else:
        idx_bigger = f.idx 
    biggers.append(tank.fishes[idx_bigger])
    if np.min(tank3.sizes) != f2.size:
        idx_smaller = np.arange(tank.n_fish)[ranks == (l_rank - 1)][0]
    else:
        idx_smaller = f2.idx
    smallers.append(tank2.fishes[idx_smaller])
    losers.append(f2)

    match1 = Fish(size = f.size,effort_method=params.effort_method)
    match2 = Fish(size = f2.size,effort_method=params.effort_method)
    #print(f.estimate,f2.estimate,match1.estimate,match2.estimate)
    fight1 = Fight(f,match1,outcome_params=params.outcome_params)
    fight2 = Fight(f2,match2,outcome_params=params.outcome_params)
    fight1.run_outcome()
    fight2.run_outcome()
    #print(f2.effort,match2.effort,f.effort)
    final_win.append(1-fight1.outcome)
    final_loss.append(1-fight2.outcome)
    
print('win success:',np.mean(final_win),'sem:',np.std(final_win) / np.sqrt(len(final_win)))
print('loser success:',np.mean(final_loss),'sem:',np.std(final_loss)/np.sqrt(len(final_loss)))
print('winner estimate',np.mean([f.estimate for f in winners]))
print('loser estimate',np.mean([f.estimate for f in losers]))
print('match estimate:',match1.estimate)
#fig2,ax2 = plt.subplots()
#for f in fishes:
n_rounds = params.n_fights * (params.n_fish-1)+1
xs = np.arange(n_rounds-1,n_rounds*2)
win_pre, win_post = [],[]
loser_pre,loser_post = [],[]
est_pre,est_post = [],[]

est_bigs,est_smalls = [],[]

ax.plot([0,0],[0,1],color='gold',label='winners')
ax.plot([0,0],[0,1],color='darkblue',label='losers')
for f_i in range(len(winners)):
    f = winners[f_i]
    pre_success = np.mean(np.array(f.win_record)[:n_rounds-1,1])
    post_success = np.mean(np.array(f.win_record)[n_rounds:,1])
    pre_estimate = np.mean(np.array(f.est_record)[:n_rounds-1])
    post_estimate = np.mean(np.array(f.est_record)[n_rounds:])
    win_pre.append(pre_success)
    win_post.append(post_success)
    est_pre.append(pre_estimate)
    est_post.append(post_estimate)
    #f = tank.fishes[0]
    ax.plot(np.array(f.est_record)-f.est_record[n_rounds-1],color='gold',alpha=.1)
    jitter = (np.random.rand() - .5) * .01
    #ax.plot(np.array(f.win_record)[:,1] + jitter,color='green',alpha=.01)
    if biggers[f_i].idx != f.idx:
        est_diff = biggers[f_i].size-f.est_record[n_rounds-1]
        est_bigs.append(est_diff) ## Careful...
        #ax.axhline(biggers[f_i].size-f.est_record[n_rounds-1],color='red',alpha=.1)
        #ax.axhline(biggers[f_i].est_record[n_rounds + 1]-f.est_record[n_rounds-1],color='blue',alpha=.1)
for f_i in range(len(losers)):
    f = losers[f_i]
    pre_success = np.mean(np.array(f.win_record)[:n_rounds-1,1])
    post_success = np.mean(np.array(f.win_record)[n_rounds:,1])
    loser_pre.append(pre_success)
    loser_post.append(post_success)
    ys = np.array(f.est_record) - f.est_record[n_rounds-1]
    ax.plot(ys,color='darkblue',alpha=.1)
    if smallers[f_i].idx != f.idx:
        est_diff = smallers[f_i].size-f.est_record[n_rounds-1]
        est_smalls.append(est_diff) ## Careful...

    #print(f.est_record)

if False: ## Show the size to next biggest and next smallest
    ax.axhline(np.mean(est_bigs),color='gray')
    ax.axhline(np.mean(est_smalls),color='gray')
## Calculate probability as a function of steps: 
win_array = np.array([f.win_record for f in winners])[:,:,1]
loser_array =np.array([f.win_record for f in losers])[:,:,1]

avg_winners = np.mean(win_array,axis=0)
avg_losers = np.mean(loser_array,axis=0)

print('before after win:')
print(np.mean(win_pre),np.mean(win_post))
print(np.mean(est_pre),np.mean(est_post))
print('before after loss:')
print(np.mean(loser_pre),np.mean(loser_post))

fig1,ax1 = plt.subplots()

ax1.plot(avg_winners)
ax1.plot(avg_losers)
ax1.axhline(.5,color='black',linestyle=':')
fig1.show()

#print('pre post win rate:')
#print(win_pre,win_post)
#print(loser_pre,loser_post)


ax.axvline(params.n_fights * (params.n_fish-1),color='black',linestyle=':',label='forced win/loss')
ax.axhline(0,color='black',label='estimate prior to staged fight')
ax.set_xlim([n_rounds-6.5,n_rounds+14.5])
ax.set_xticks(np.arange(15,37,5))
ax.set_ylim([-10.5,10.5])
ax.legend(loc='upper right')
ax.set_ylabel('Normalized size estimate')
ax.set_xlabel('Contest number')

fig.tight_layout()
fig.savefig('./imgs/win_loss.png',dpi=300)
fig.savefig('./imgs/win_loss.svg')

print('Done!')
