#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation
from params import Params

import numpy as np
from scipy.stats import binom_test,norm
from scipy.stats import f_oneway,pearsonr

from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy
import itertools

#s,e,l = .6,.3,.1

params = Params()
if True:
    params.effort_method = 'SmoothPoly'

params.poly_param_a = 2
params.poly_param_c = 0.1
params.n_fights = 50 
#params.size = 50
params.energy_cost = False
params.acuity = 0.1
#params.awareness = 0.2
params.start_energy = 1

params.iterations = 100
params.n_fish = 5
params.n_fights = 3
params.f_method = 'balanced'
params.update_method = 'bayes'
params.f_outcome = 'math'
params.set_L()



## Have a fish win-lose, or lose-win, then measure the winner effect for that fish

## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
scale = 2

upset_count = 0
win_count1,win_count2 = 0,0
win_shifts,loss_shifts = [],[]
## Lets assume their estimate is correct.

params.age = 51
#params.size = 50 

## Set up the various fish conditions
#size_params = params.copy() ## Fish determine prior from their size
self_params = params.copy() ## fish determine prior from their size/age
self_params.awareness = 0.2
self_params.set_params()

big_params = params.copy() ## Fish determine prior from surround fish (controlled to be big)
big_prior = norm.pdf(params.xs,60,5)
big_params.prior = big_prior / sum(big_prior)

small_params = params.copy() ## Fish determine prior from surrounding fish (controlled to be small)
small_prior = norm.pdf(params.xs,40,5)
small_params.prior = small_prior / sum(small_prior)

null_params = params.copy() ## Fish have null prior
guess_params = params.copy() ## Fish have random prior (but confident)

#guess_params.awareness = 20
#guess_params.awareness = 0.5

opp_params = params.copy()
opp_params.prior = True
opp_params.size = 50

## Uniform prior
null_params.prior = np.ones_like(null_params.xs) / len(null_params.xs)

self_fishes = [Fish(n+1,self_params) for n in range(params.iterations)]
null_fishes = [Fish(n+1,null_params) for n in range(params.iterations)]
big_fishes = [Fish(n+1,big_params) for n in range(params.iterations)]
small_fishes = [Fish(n+1,small_params) for n in range(params.iterations)]
guess_fishes = []
## Make sloppy fish confident
for n in tqdm(range(params.iterations)):
    guess_prior = norm.pdf(params.xs,np.random.random() * 99 + 1,10)
    guess_params.prior = guess_prior / sum(guess_prior)
    f = Fish(n+1,guess_params)
    #prior = f.prior ** 10
    #f.prior = prior / np.sum(prior)
    f.get_stats()
    guess_fishes.append(f)
    null_fishes[n].get_stats()

for n in tqdm(range(params.iterations)):
    f = Fish(n+1,guess_params)
    #print(null_fishes[n].estimate,null_fishes[n].size)
opp_fish = Fish(0,opp_params)

#print(opp_fish.estimate)
#print(prior_fishes[0].estimate,null_fishes[0].estimate,guess_fishes[0].estimate)
p_efforts,n_efforts,g_efforts = [],[],[] # Prior(self), uNiform, random (Guess)
b_efforts,s_efforts = [],[] ## big adjacent, small adjactent

effort_array = np.empty([params.iterations,5,3])


for n in tqdm(range(params.iterations)):
    p,o = self_fishes[n],null_fishes[n] ## not sure why I use o vs n for null here
    g = guess_fishes[n]
    b,s = big_fishes[n],small_fishes[n]

    self_fight = Fight(self_fishes[n],opp_fish)
    null_fight = Fight(null_fishes[n],opp_fish)
    guess_fight = Fight(guess_fishes[n],opp_fish)
    big_fight = Fight(big_fishes[n],opp_fish)
    small_fight = Fight(small_fishes[n],opp_fish)

    for m in range(params.n_fights):
        self_fight.run_outcome()
        null_fight.run_outcome()
        guess_fight.run_outcome()
        big_fight.run_outcome()
        small_fight.run_outcome()

        effort_array[n,0,m] = p.effort
        effort_array[n,1,m] = o.effort
        effort_array[n,2,m] = g.effort
        effort_array[n,3,m] = b.effort
        effort_array[n,4,m] = s.effort
    p_efforts.append(p.effort)
    n_efforts.append(o.effort)
    g_efforts.append(g.effort)
    b_efforts.append(b.effort)
    s_efforts.append(s.effort)
    #print(o.size,o.effort)
p_mean,p_std = np.mean(p_efforts),np.std(p_efforts)
n_mean,n_std = np.mean(n_efforts),np.std(n_efforts)
g_mean,g_std = np.mean(g_efforts),np.std(g_efforts)
print(np.mean(p_efforts),np.mean(n_efforts),np.mean(g_efforts))
print(np.std(p_efforts),np.std(n_efforts),np.std(g_efforts))

print('self-informed mean:',np.mean(effort_array[:,0,:],axis=0))

#fig,(ax,ax2,ax3) = plt.subplots(3)

print('Uniform ONE WAY ANOVA:')
print(f_oneway(*effort_array[:,1]))


print('Random ONE WAY ANOVA:')
print(f_oneway(*effort_array[:,2]))

print('SELF ONE WAY ANOVA:')
print(f_oneway(*effort_array[:,0]))
print('Small ONE WAY ANOVA:')
print(f_oneway(*effort_array[:,4]))
print('bigger ONE WAY ANOVA:')
print(f_oneway(*effort_array[:,3]))

fish_groups = [self_fishes,null_fishes,guess_fishes,big_fishes,small_fishes]
strats = ['Self-Informed','Uniform','Random','Big-informed','Small-informed']

for strat in range(5):
    ys = np.mean(effort_array[:,strat],axis=1)
    fishes = fish_groups[strat]
    xs = [fishes[f].size for f in range(params.iterations)]
    print(strats[strat],pearsonr(xs,ys))

fig,ax = plt.subplots()
#ax.bar([1,2,3],[p_mean,n_mean,g_mean],yerr=[p_std,n_std,g_std],alpha=.1)
p_norms = np.mean(effort_array[:,0],axis=1)
n_norms = np.mean(effort_array[:,1],axis=1)
g_norms = np.mean(effort_array[:,2],axis=1)
b_norms = np.mean(effort_array[:,3],axis=1)
s_norms = np.mean(effort_array[:,4],axis=1)

p_sizes = [self_fishes[f].size/100+2.0 for f in range(params.iterations)]
n_sizes = [null_fishes[f].size/100+0.0 for f in range(params.iterations)]
g_sizes = [guess_fishes[f].size/100+1 for f in range(params.iterations)]
b_sizes = [big_fishes[f].size/100+3.25 for f in range(params.iterations)]
s_sizes = [small_fishes[f].size/100+2.75 for f in range(params.iterations)]

ax.boxplot([n_norms,g_norms,p_norms,s_norms,b_norms],positions=[0.5,1.5,2.5,3.25,3.72],widths=0.1)

cors = ['darkgray','dimgray','royalblue','darkblue','darkblue']

ax.scatter(p_sizes,p_norms,alpha=0.3,color=cors[2])
ax.scatter(n_sizes,n_norms,alpha=0.3,color=cors[0])
ax.scatter(g_sizes,g_norms,alpha=0.3,color=cors[1])
ax.scatter(b_sizes,b_norms,alpha=0.3,color=cors[4])
ax.scatter(s_sizes,s_norms,alpha=0.3,color=cors[3])

#ax.scatter([self_fishes[f].size/100+0.5 for f in range(params.iterations)],np.mean(effort_array[:,1],axis=1))

ax.set_xticklabels(['Uniform\nPrior','Random\nPrior','Self-Informed\nPrior','Small-Informed\nPrior','Big-Informed\nPrior'],rotation=45)
ax.axvline(2.0,linestyle=':',color='black')
#ax2.scatter([f.guess for f in prior_fishes],[f.effort for f in prior_fishes])
#ax3.scatter([f.estimate for f in prior_fishes],[f.effort for f in prior_fishes])
ax.set_ylabel('Effort')
fig.tight_layout()
fig.show()
plt.show()
