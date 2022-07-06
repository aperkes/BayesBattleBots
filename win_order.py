#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from scipy.stats import binom_test

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy

s,e,l = .6,.3,.01

params = SimParams()
params.effort_method = [1,1]
params.n_fights = 50 
params.n_iterations = 15
params.n_fish = 5
params.f_method = 'balanced'
params.u_method = 'bayes'
params.f_outcome = 'math'
params.outcome_params = [s,e,l]
s = Simulation(params)

winners,losers = [],[]
PLOT = 'estimate'

## Have a fish win-lose, or lose-win, then measure the winner effect for that fish

## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
iterations = 1000
scale = 2

upset_count = 0
win_count1,win_count2 = 0,0
win_shifts,loss_shifts = [],[]
## Lets assume their estimate is correct.

## Build copies of f0 with naive priors
def build_fish(idx,f0):
        return Fish(idx,size=f0.size,prior=True,effort_method=params.effort_method,fight_params=params.outcome_params,update_method=params.u_method,likelihood=f0.naive_likelihood)

def check_success(f,f_match):
    fight = Fight(f,f_match,outcome_params=params.outcome_params)
    fight.run_outcome()
    return fight1.outcome

CALC = True
age = 50
match1_results = []


if CALC:
    f0 = Fish(1,age=age,prior=True,effort_method=params.effort_method,fight_params=params.outcome_params,update_method=params.u_method)
    for i in range(iterations):
#for i in tqdm(range(iterations)):
        f = build_fish(1,f0)
        f_match = build_fish(0,f0)
        match1_results.append(check_success(f,f_match))
        f_w = copy.deepcopy(f) 
        f_l = copy.deepcopy(f)
        f2 = Fish(2,size=f1.size,prior=True,effort_method=params.effort_method,update_method=params.u_method,fight_params=params.outcome_params,likelihood=f0.naive_likelihood)

        #print(f1.size,f1.estimate)
## This fish needs to be slightly bigger, and it's estimate needs to shift up accordingly
        f_bigger = Fish(3,size=f1.size, prior=True,effort_method=params.effort_method,update_method=params.u_method,fight_params=params.outcome_params,likelihood=f0.naive_likelihood)
        f_smaller = Fish(4,size=f1.size, prior=True,effort_method=params.effort_method,update_method=params.u_method,fight_params=params.outcome_params,likelihood=f0.naive_likelihood)

        f_match1 = Fish(0,size=f1.size, prior=True,effort_method=params.effort_method,update_method=params.u_method,fight_params=params.outcome_params,likelihood=f0.naive_likelihood)
        f_match2 = Fish(0,size=f1.size, prior=True,effort_method=params.effort_method,update_method=params.u_method,fight_params=params.outcome_params,likelihood=f0.naive_likelihood)

## How often would they lose this way? 
        c_loss = Fight(f1,f_bigger,outcome=params.f_outcome,outcome_params=params.outcome_params)
        c_win = Fight(f2,f_smaller,outcome=params.f_outcome,outcome_params=params.outcome_params)
        c_loss.run_outcome()
        c_win.run_outcome()

        c_win.winner.update(True,c_win)
        c_win.loser.update(False,c_win)

        c_loss.winner.update(True,c_loss)
        c_loss.loser.update(False,c_loss)

        loss_shifts.append(c_loss.loser.estimate / c_loss.loser.est_record[0])
        win_shifts.append(c_win.winner.estimate / c_loss.loser.est_record[0])
        if c_win.winner.idx != 4:
            upset_count += 1
        if c_loss.winner.idx != 3:
            upset_count += 1

        sa_effort = c_win.winner.choose_effort(c_win.winner,strategy=[1,0])
        ma_effort = c_win.winner.choose_effort(c_win.winner,strategy=[1,1])
        #print(c_win.winner.size,c_win.winner.estimate,sa_effort,ma_effort)

        c_match1 = Fight(c_win.winner,f_match1,outcome=params.f_outcome,outcome_params=params.outcome_params)
        c_match2 = Fight(c_loss.loser,f_match2,outcome=params.f_outcome,outcome_params=params.outcome_params)

        c_match1.run_outcome()
        c_match2.run_outcome()
        if c_match1.winner.idx != 0:
            win_count1 += 1
        if c_match2.winner.idx != 0:
            win_count2 += 1
else:
    win_count1,win_count2 = 755,165 #also from previous run, saves a lot of time
    f0 = Fish(1,size=65,prior=True,effort_method=params.effort_method,fight_params=params.outcome_params,update_method=params.u_method)
    f1 = Fish(1,size=f0.size,prior=True,effort_method=params.effort_method,fight_params=params.outcome_params,update_method=params.u_method)
    c_match1 = Fight(f0,f1,outcome=0,outcome_params=params.outcome_params)
    c_match1.run_outcome()

print('winner:',win_count1,'/',iterations)
print(binom_test(win_count1,n=1000))
print('loser:',win_count2,'/',iterations)
print(binom_test(win_count2,n=1000))
if CALC:
    print('win shift:',np.mean(win_shifts)-1)
    print('loss shift:',np.mean(loss_shifts)-1)
fig,ax = plt.subplots()

sa_win,sa_loss = 643,278 # these numbers are pulled from a previous trial, change the effort method in line 24 to [1,0] to generate them
ax.bar([1,2,5,6],[win_count1,win_count2,sa_win,sa_loss],color=['gold','darkblue','gold','darkblue'])
ax.set_xticks([1,2,5,6])
ax.set_xticklabels(['win','loss','win','loss'])
ax.axhline(500,linestyle=':',color='black')

ax.set_ylabel('n_wins')
ax.set_ylim([0,1000])
#fig.show()

fig1,ax1 = plt.subplots()

#ax1.plot(f0.xs,f0._define_naive_likelihood())
win_likelihood = f0._define_likelihood_mutual(c_match1,True)
loss_likelihood = f0._define_likelihood_mutual(c_match1,False)
win_estimate = np.sum(win_likelihood * f0.xs / np.sum(win_likelihood))
loss_estimate = np.sum(loss_likelihood * f0.xs / np.sum(loss_likelihood))

print(f0.size)
print(win_estimate-f0.size,loss_estimate-f0.size)
ax1.plot(f0.xs,f0._define_likelihood_mutual(c_match1,True))
ax1.plot(f0.xs,f0._define_likelihood_mutual(c_match1,False))
ax1.plot(f0.size + (f0.size-f0.xs),f0._define_likelihood_mutual(c_match1,True),alpha=.5)
ax1.axvline(f0.size,linestyle=':',color='black')
ax1.axhline(0.50,color='gray')
#fig1.show()

fig2,ax2 = plt.subplots()
ax2.plot(c_win.winner.xs,c_win.winner.prior,color='gold')
ax2.plot(c_loss.loser.xs,c_loss.loser.prior,color='darkblue')
ax2.axvline(c_win.winner.size,color='black')
ax2.plot(f0.xs,f0.prior/np.sum(f0.prior),color='black',alpha=.7,linestyle=':')
print('winner invests:',c_win.winner.choose_effort(c_loss.loser))
print('loser invests:',c_loss.loser.choose_effort(c_win.winner))
print('naive invests:',f0.choose_effort(c_win.loser))
print(c_win.winner.size,c_loss.loser.size,f0.size)

plt.show()
