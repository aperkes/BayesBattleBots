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

s,e,l = .6,.3,.01

max_fights = 20

params_bayes = Params()
params_bayes.iterations = 400
params_bayes.effort_method = 'SmoothPoly'
params_bayes.n_fights = 5
params_bayes.n_fish = 5
params_bayes.f_method = 'balanced'
params_bayes.f_outcome = 'math'
params_bayes.outcome_params = [s,e,l]
params_bayes.set_L()
params_bayes.u_method = 'bayes'

#params_bayes.mean_size = 58.5

params_boost = params_bayes.copy()
params_boost.u_method = 'size_boost'

npc_params = params_bayes.copy()
npc_params.baseline_effort = None ## use random effort
#fig2,ax2 = plt.subplots()

params_fixed = params_bayes.copy()
params_fixed.update_method = None

params_linear = params_bayes.copy()
params_linear.poly_param_b = 1
params_linear.poly_param_m = 0

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

params = params_linear

fishes = [Fish(0,params) for f in range(params.iterations)]
opp_fishes = [Fish(i + 2) for i in range(max_fights)]
matched_fishes = [Fish(1,f.params) for f in fishes]

win_props = np.empty([max_fights,params.iterations])
loss_props = np.empty_like(win_props)

for i in tqdm(range(params.iterations)):
    f = fishes[i]
    matched_fish = matched_fishes[i]
    for r in range(max_fights):
        opp_fish = opp_fishes[r]
        focal_winner = copy.deepcopy(f)
        focal_loser = copy.deepcopy(f)

        win_fight = Fight(matched_fish,focal_winner,params,outcome=0)
        lose_fight = Fight(matched_fish,focal_loser,params,outcome=1)
        
        win_fight.run_outcome()
        lose_fight.run_outcome()
        focal_winner.update(1,win_fight)
        focal_loser.update(0,lose_fight)

        check_post_win = Fight(opp_fish,focal_winner,params)
        win_props[r,i] = check_post_win.run_outcome()

        check_post_loss = Fight(opp_fish,focal_loser,params)
        loss_props[r,i] = check_post_loss.run_outcome()

        real_fight = Fight(matched_fish,f,params)
        real_outcome = real_fight.run_outcome()
        f.update(real_outcome,real_fight)

fig,ax = plt.subplots()

ax.axhline(0.5,color='black',linestyle=':')

xs = np.arange(0,max_fights)

if False:
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

linear_loss_ys = np.mean(loss_props,1)
linear_loss_errs = np.std(loss_props,1) / np.sqrt(params.iterations)

ax.plot(xs,linear_loss_ys,label='linear_losers',color='green')
ax.fill_between(xs,linear_loss_ys-linear_loss_errs,linear_loss_ys + linear_loss_errs,color='gray',alpha=0.5)
#import pdb
#pdb.set_trace()

if params_bayes.plot_me:
    plt.show()
if params_bayes.save_me:
    fig.savefig('./figures/fig6b.png',dpi=300)

print('done!')
