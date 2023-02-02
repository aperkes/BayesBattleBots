#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish,FishNPC
from fight import Fight
from tank import Tank
from params import Params

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy

## A function that pre-runs a set number of fights, then measures winner effect
def run_tanks(params=Params(),pre_rounds=0,progress=False):
    drop_count = 0
    winners,losers = [],[]
## Let fish duke it out, then pull a fish out, check it's success against a size matched fish, and put it back in
    results = [[],[]]
    params.n_fights = pre_rounds
    if progress:
        iterator = tqdm(range(params.iterations))
    else:
        iterator = range(params.iterations)
    for i in iterator:
        focal_fish = Fish(0,params)
        matched_fish = FishNPC(npc_params)
        npc_fish = [FishNPC(f,npc_params) for f in range(pre_rounds + 1)]

        for f in range(pre_rounds):
            fight = Fight(focal_fish,npc_fish[f],params)
            fight.run_outcome()
            focal_fish.update(1-fight.outcome,fight)

        if focal_fish.estimate < 8 or focal_fish.estimate > 99:
            drop_count += 1
            pass
            #continue
        matched_fish.size = focal_fish.size
        match = Fight(matched_fish,focal_fish,params,outcome=np.random.randint(2))
        outcome = match.run_outcome()
        focal_fish.update(outcome,match)
        #print(focal_fish.effort,focal_fish.energy)

        match2 = Fight(matched_fish,focal_fish,params)
        test_outcome = match2.run_outcome()
        if outcome == 1:
            winners.append(focal_fish)
        else:
            losers.append(focal_fish)
        results[outcome].append(test_outcome)
    #print('dropped n crappy fish:',drop_count)
    return winners,losers,results

s,e,l = .6,.3,.01

max_fights = 20

params_bayes = Params()
params_bayes.iterations = 100
params_bayes.effort_method = 'SmoothPoly'
params_bayes.n_fights = 5
params_bayes.n_fish = 5
params_bayes.f_method = 'balanced'
params_bayes.f_outcome = 'math'
params_bayes.outcome_params = [s,e,l]
params_bayes.set_L()
params_bayes.u_method = 'bayes'

params_boost = params_bayes.copy()
params_boost.u_method = 'size_boost'

npc_params = params_bayes.copy()
npc_params.baseline_effort = None ## use random effort
#fig2,ax2 = plt.subplots()

params_fixed = params_bayes.copy()
params_fixed.update_method = None

params_linear = params_bayes.copy()
params_linear.update_method = 'linear'

params_bayes.plot_me = True
params_bayes.save_me = False


xs = np.arange(max_fights)
win_ys,win_errs = np.zeros(max_fights),np.zeros(max_fights)
loss_ys,loss_errs = np.zeros_like(win_ys),np.zeros_like(win_errs)

fixed_win_ys,fixed_win_errs = np.zeros_like(win_ys),np.zeros_like(win_errs)
fixed_loss_ys,fixed_loss_errs = np.zeros_like(win_ys),np.zeros_like(win_errs)

linear_win_ys,linear_win_errs = np.zeros_like(win_ys),np.zeros_like(win_errs)
linear_loss_ys,linear_loss_errs = np.zeros_like(win_ys),np.zeros_like(win_errs)

for r in range(max_fights):
    #winners_naive,losers_naive,results_naive = run_tanks(params=params_bayes,pre_rounds=0)
    print('running for',r)
    winners_exp,losers_exp,results_exp = run_tanks(params=params_bayes,pre_rounds=r)
    winners_fixed,losers_fixed,results_fixed = run_tanks(params=params_fixed,pre_rounds=r)
    winners_linear,losers_linear,results_linear = run_tanks(params=params_linear,pre_rounds=r)

    win_ys[r] = np.mean(results_exp[1])
    win_errs[r] = np.std(results_exp[1]) / np.sqrt(len(winners_exp))

    loss_ys[r] = np.mean(results_exp[0])
    loss_errs[r] = np.std(results_exp[0]) / np.sqrt(len(losers_exp))

    fixed_win_ys[r] = np.mean(results_fixed[1])
    fixed_win_errs[r] = np.std(results_fixed[1]) / np.sqrt(len(winners_fixed))

    fixed_loss_ys[r] = np.mean(results_fixed[0])
    fixed_loss_errs[r] = np.std(results_fixed[0]) / np.sqrt(len(losers_fixed))

    linear_win_ys[r] = np.mean(results_linear[1])
    linear_win_errs[r] = np.std(results_linear[1]) / np.sqrt(len(winners_linear))

    linear_loss_ys[r] = np.mean(results_linear[0])
    linear_loss_errs[r] = np.std(results_linear[0]) / np.sqrt(len(losers_linear))
fig,ax = plt.subplots()

ax.axhline(0.5,color='black',linestyle=':')

ax.plot(xs,win_ys,label='bayes_winners',color='gold')
ax.fill_between(xs,win_ys-win_errs,win_ys+win_errs,color='gray',alpha=0.5)

ax.plot(xs,loss_ys,label='bayes_losers',color='blue')
ax.fill_between(xs,loss_ys-loss_errs,loss_ys + loss_errs,color='gray',alpha=0.5)

ax.plot(xs,fixed_win_ys,label='fixed_winners',color='gray')
ax.fill_between(xs,fixed_win_ys-fixed_win_errs,fixed_win_ys + fixed_win_errs,color='gray',alpha=0.5)

ax.plot(xs,fixed_loss_ys,label='fixed_losers',color='black')
ax.fill_between(xs,fixed_loss_ys-fixed_loss_errs,fixed_loss_ys + fixed_loss_errs,color='gray',alpha=0.5)

ax.plot(xs,linear_win_ys,label='linear_winners',color='green')
ax.fill_between(xs,linear_win_ys-linear_win_errs,linear_win_ys + linear_win_errs,color='gray',alpha=0.5)

ax.plot(xs,linear_loss_ys,label='linear_losers',color='green')
ax.fill_between(xs,linear_loss_ys-linear_loss_errs,linear_loss_ys + linear_loss_errs,color='gray',alpha=0.5)

print('n winners, n_losers')
print(len(results_linear[0]),len(results_linear[1]))

for l in losers_linear:
    print(l.est_record[-2:])
    print(l.estimate)

#import pdb
#pdb.set_trace()

if params_bayes.plot_me:
    plt.show()
if params_bayes.save_me:
    fig.savefig('./figures/fig6b.png',dpi=300)

print('done!')
