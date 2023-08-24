#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation
from params import Params

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.stats import ttest_ind

from tqdm import tqdm

import copy
import cProfile,pstats

from joblib import Parallel, delayed

## for debugging runtime warnings:
#np.seterr(all='raise')

#profile = cProfile.Profile()
#profile.enable()

def run_experiment(params):
    fishes = [Fish(f,params) for f in range(params.n_fish)]
    tank = Tank(fishes,params)
    if INIT:
        tank._initialize_likelihood()
    tank.run_all(False)

    f = tank.fishes[0].copy()
    f2 = tank.fishes[1].copy()
    opp = Fish(size = f.size*scale)
    opp2 = Fish(size = f2.size/scale)
    f_win = Fight(f,opp,params,outcome=0)
    f_loss = Fight(f2,opp2,params,outcome=1) 
    f_win.winner = f
    f_win.loser = opp
    f_loss.winner = opp2
    f_loss.loser = f2

    ## Because this fight is never actually run, it never asks about effort, so you have to provide it
    f_win.winner.effort = 0.5
    f_win.loser.effort = 0.5
    f_loss.loser.effort = 0.5
    f_loss.winner.effort = 0.5

    f.update(True,f_win)    
    f2.update(False,f_loss)

    fishes2 = [fishy.copy() for fishy in fishes]
    fishes2[0] = f
    tank2 = Tank(fishes2,params)

    fishes3 = [fishy.copy() for fishy in fishes]
    fishes3[1] = f2
    tank3 = Tank(fishes3,params)

    tank2.run_all(False)
    tank3.run_all(False)

    winner_r = f
    rank_order= np.argsort(np.array(tank2.sizes))
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(tank2.n_fish) ## This took me an embarrasingly long time to get right. 
    f_rank = ranks[f.idx]
    l_rank = ranks[f2.idx]

    winner_record_r = np.sum(tank2.history[-1,f.idx]) / (params.n_fish - 1)
    loser_record_r = np.sum(tank3.history[-1,f2.idx]) / (params.n_fish - 1)

    if np.max(tank2.sizes) != f2.size:
        idx_bigger_l = np.arange(tank.n_fish)[ranks == (l_rank + 1)][0]
        loser_v_bigger_r = tank3.win_record[f2.idx,idx_bigger_l]/params.n_rounds
    else:
        loser_v_bigger_r = np.nan
    if np.min(tank2.sizes) != f.size:
        idx_smaller_w = np.arange(tank.n_fish)[ranks == (f_rank - 1)][0]
        winner_v_smaller_r = tank2.win_record[f.idx,idx_smaller_w]/params.n_rounds
    else:
        winner_v_smaller_r = np.nan
    
    if np.max(tank2.sizes) != f.size:
        idx_bigger = np.arange(tank.n_fish)[ranks == (f_rank + 1)][0]
        winner_v_bigger_r = tank2.win_record[f.idx,idx_bigger]/params.n_rounds ## should this be rounds or fights.

    else:
        idx_bigger = f.idx 
        winner_v_bigger_r = np.nan

    biggers_r = tank.fishes[idx_bigger]

    if np.min(tank3.sizes) != f2.size:
        idx_smaller = np.arange(tank.n_fish)[ranks == (l_rank - 1)][0]
        loser_v_smaller_r = tank3.win_record[f2.idx,idx_smaller]/params.n_fights
    else:
        idx_smaller = f2.idx
        loser_v_smaller_r = np.nan

    smaller_r = tank2.fishes[idx_smaller]
    loser_r = f2

    results = [winner_v_bigger_r, loser_v_bigger_r,
                winner_v_smaller_r, loser_v_smaller_r,
                winner_r, loser_r]
    return results

def run_simulation(params):
    winners = [np.nan] * replicates
    losers = [np.nan] * replicates
    biggers = [np.nan] * replicates
    smallers = [np.nan] * replicates

## So for each replicate, we fill in 
    winner_v_bigger = np.empty(replicates)
    loser_v_bigger = np.empty_like(winner_v_bigger)

    winner_v_smaller = np.empty_like(winner_v_bigger)
    loser_v_smaller = np.empty_like(winner_v_bigger)

    winner_record = np.empty_like(winner_v_bigger)
    loser_record = np.empty_like(winner_v_bigger)

    winner_v_bigger.fill(np.nan)
    winner_v_smaller.fill(np.nan)
    loser_v_bigger.fill(np.nan)
    loser_v_smaller.fill(np.nan)
    winner_record.fill(np.nan)
    loser_record.fill(np.nan)

    for r in range(replicates):
        results = run_experiment(params)

        winner_v_bigger[r],loser_v_bigger[r] = results[0:2]
        winner_v_smaller[r],loser_v_smaller[r] = results[2:4]
        winners[r],losers[r] = results[4:]

    win_outcomes = np.empty(replicates)
    loss_outcomes = np.empty_like(win_outcomes) 

    for f_i in range(len(winners)):
        f = winners[f_i]
        check_fish = Fish(0,params,size=f.size)
        check_fight = Fight(check_fish,f,params)
        outcome = check_fight.run_outcome()
        win_outcomes[f_i] = outcome

    for f_i in range(len(losers)):
        f = losers[f_i]

### Check loser against a size matched fish
        check_fish = Fish(0,params,size=f.size)
        check_fight = Fight(check_fish,f,params)
        outcome = check_fight.run_outcome()
        loss_outcomes[f_i] = outcome
       
    cleaned_wvs = winner_v_smaller[~np.isnan(winner_v_smaller)]
    cleaned_lvs = loser_v_smaller[~np.isnan(loser_v_smaller)]
    cleaned_wvb = winner_v_bigger[~np.isnan(winner_v_bigger)]
    cleaned_lvb = loser_v_bigger[~np.isnan(loser_v_bigger)]

    t_s,p_s = ttest_ind(cleaned_wvs,cleaned_lvs)
    t_b,p_b = ttest_ind(cleaned_wvb,cleaned_lvb)
    t_m,p_m = ttest_ind(win_outcomes,loss_outcomes)

    smaller_diff = winner_v_smaller - loser_v_smaller
    s_diff = np.nanmean(smaller_diff)
    s_diff_std = np.nanstd(smaller_diff)

    bigger_diff = winner_v_bigger - loser_v_bigger
    b_diff = np.nanmean(bigger_diff)
    b_diff_std = np.nanstd(bigger_diff)

    m_diff_array = win_outcomes - loss_outcomes
    m_diff = np.nanmean(m_diff_array)
    m_diff_std = np.nanstd(m_diff_array)

    #ps_array[rep,i] = p_s
    #pb_array[rep,i] = p_b
    #pm_array[rep,i] = p_m

    #smean_array[rep,i] = s_diff
    #bmean_array[rep,i] = b_diff
    #mmean_array[rep,i] = m_diff

    #mstd_array[rep,i] = m_diff_std
    #sstd_array[rep,i] = s_diff_std
    #bstd_array[rep,i] = b_diff_std 

    if b_diff > 1 or s_diff > 1 or m_diff > 1:
        print('uh oh.')

    m_results = [p_m,m_diff,m_diff_std]
    s_results = [p_s,s_diff,s_diff_std]
    b_results = [p_b,b_diff,b_diff_std]

    tri_results = [m_results,s_results,b_results]
    return tri_results

#s,e,l = -.8,-.6,-0.99
s,e,l=-0.9,-0.5,-0.7

params = Params()
params.xs = np.linspace(1,100,100)
params.mean_size = 50
params.sd_size = 20
#params.poly_param_c = 0
#params.effort_method = [1,1]
params.n_fights = 5 
params.n_fish = 5
params.acuity = 5
params.f_method = 'balanced'
#params.u_method = 'bayes'
#params.f_outcome = 'math'
params.outcome_params = [s,e,l]
params.set_params()
## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
#replicates = 10
scale = 1

#plt.rcParams.update({'font.size': 18})
#plt.rcParams.update({'lines.linewidth':5})
#plt.rcParams.update({'axes.linewidth':2})

INIT = False

#ew_pairs = []

iterations = 100
min_reps = 5
max_reps = 8
#rep_array = np.arange(min_reps,max_reps)
rep_array = np.array([5,25,40,60,80,100,125,200,300,500])
r_bins = len(rep_array)

ps_array = np.empty([r_bins,iterations])
pb_array = np.empty_like(ps_array)
pm_array = np.empty_like(ps_array)

smean_array = np.empty_like(ps_array)
bmean_array = np.empty_like(ps_array)
mmean_array = np.empty_like(ps_array)

mstd_array = np.empty_like(ps_array)
sstd_array = np.empty_like(ps_array)
bstd_array = np.empty_like(ps_array)

print('n iterations:',iterations)


for rep in range(r_bins):
    replicates = rep_array[rep]
    print('running with',rep_array[rep],'replicates')

## This code was way to slow. This speeds it up by around 10x
    all_iterations = Parallel(n_jobs=-1)(delayed(run_simulation)(params) for _ in tqdm(range(iterations)))
    for i in range(iterations):
        #tri_results = run_simulation(params)
        tri_results = all_iterations[i]
        m_results,s_results,b_results = tri_results
        pm_array[rep,i],mmean_array[rep,i],mstd_array[rep,i] = m_results
        ps_array[rep,i],smean_array[rep,i],sstd_array[rep,i] = s_results
        pb_array[rep,i],bmean_array[rep,i],bstd_array[rep,i] = b_results

"""
        winners = [np.nan] * replicates
        losers = [np.nan] * replicates
        biggers = [np.nan] * replicates
        smallers = [np.nan] * replicates

## So for each replicate, we fill in 
        winner_v_bigger = np.empty(replicates)
        loser_v_bigger = np.empty_like(winner_v_bigger)

        winner_v_smaller = np.empty_like(winner_v_bigger)
        loser_v_smaller = np.empty_like(winner_v_bigger)

        winner_record = np.empty_like(winner_v_bigger)
        loser_record = np.empty_like(winner_v_bigger)

        winner_v_bigger.fill(np.nan)
        winner_v_smaller.fill(np.nan)
        loser_v_bigger.fill(np.nan)
        loser_v_smaller.fill(np.nan)
        winner_record.fill(np.nan)
        loser_record.fill(np.nan)

## This loop is just too slow, needed to optimize it
## The syntax is weird, but it parallelizes the experiment across CPUs
#        all_results = Parallel(n_jobs=12)(delayed(run_experiment)(params) for _ in range(replicates))
        for r in range(replicates):
            results = run_experiment(params)
            #results = all_results[r]

            winner_v_bigger[r],loser_v_bigger[r] = results[0:2]
            winner_v_smaller[r],loser_v_smaller[r] = results[2:4]
            winners[r],losers[r] = results[4:]

        win_outcomes = np.empty(replicates)
        loss_outcomes = np.empty_like(win_outcomes) 

        for f_i in range(len(winners)):
            f = winners[f_i]
            check_fish = Fish(0,params,size=f.size)
            check_fight = Fight(check_fish,f,params)
            outcome = check_fight.run_outcome()
            win_outcomes[f_i] = outcome

        for f_i in range(len(losers)):
            f = losers[f_i]

### Check loser against a size matched fish
            check_fish = Fish(0,params,size=f.size)
            check_fight = Fight(check_fish,f,params)
            outcome = check_fight.run_outcome()
            loss_outcomes[f_i] = outcome
           
        cleaned_wvs = winner_v_smaller[~np.isnan(winner_v_smaller)]
        cleaned_lvs = loser_v_smaller[~np.isnan(loser_v_smaller)]
        cleaned_wvb = winner_v_bigger[~np.isnan(winner_v_bigger)]
        cleaned_lvb = loser_v_bigger[~np.isnan(loser_v_bigger)]

        t_s,p_s = ttest_ind(cleaned_wvs,cleaned_lvs)
        t_b,p_b = ttest_ind(cleaned_wvb,cleaned_lvb)
        t_m,p_m = ttest_ind(win_outcomes,loss_outcomes)

        smaller_diff = winner_v_smaller - loser_v_smaller
        s_diff = np.nanmean(smaller_diff)
        s_diff_std = np.nanstd(smaller_diff)

        bigger_diff = winner_v_bigger - loser_v_bigger
        b_diff = np.nanmean(bigger_diff)
        b_diff_std = np.nanstd(bigger_diff)

        m_diff_array = win_outcomes - loss_outcomes
        m_diff = np.nanmean(m_diff_array)
        m_diff_std = np.nanstd(m_diff_array)

        ps_array[rep,i] = p_s
        pb_array[rep,i] = p_b
        pm_array[rep,i] = p_m

        smean_array[rep,i] = s_diff
        bmean_array[rep,i] = b_diff
        mmean_array[rep,i] = m_diff

        mstd_array[rep,i] = m_diff_std
        sstd_array[rep,i] = s_diff_std
        bstd_array[rep,i] = b_diff_std 

        #m_results = [pm_array,mmean_array,mstd_array]
        #s_results = [ps_array,smean_array,sstd_array]
        #b_results = [pb_array,bmean_array,bstd_array]

        #tri_results = [m_results,s_results,b_results]

        if b_diff > 1 or s_diff > 1 or m_diff > 1:
            import pdb;pdb.set_trace()
            print('uh oh.')
    """

## Things I needj
xs = rep_array
log_xs = np.log(rep_array)
log_xs = np.emath.logn(5, rep_array)

## Probabilities for size matched
prob_sig_m = pm_array < 0.05
mean_prob_m = np.nanmean(prob_sig_m,1)
std_prob_m = np.nanstd(prob_sig_m,1)

## Probabilities for smaller
prob_sig_s = ps_array < 0.05
mean_prob_s = np.nanmean(prob_sig_s,1)
std_prob_s = np.nanstd(prob_sig_s,1)

## Probabilities for bigger
prob_sig_b = pb_array < 0.05
mean_prob_b = np.nanmean(prob_sig_b,1)
std_prob_b = np.nanstd(prob_sig_b,1)

## Means (and std) for size matched, smallers, and biggers
mean_diff_m = np.nanmean(mmean_array,1)
sem_diff_m = np.nanmean(mstd_array,1) / np.sqrt(rep_array)

mean_diff_s = np.nanmean(smean_array,1)
#sem_diff_s = np.nanstd(smean_array,1) / np.sqrt(rep_array)
sem_diff_s = np.nanmean(sstd_array,1) / np.sqrt(rep_array)

mean_diff_b = np.nanmean(bmean_array,1)
#sem_diff_b = np.nanstd(bmean_array,1) / np.sqrt(rep_array)
sem_diff_b = np.nanmean(bstd_array,1) / np.sqrt(rep_array)


m_color = 'black'
s_color = 'darkblue'
b_color = 'gold'
## Plot the probability of detecting a significant difference (with std across 1000 iterations)

fig,ax = plt.subplots()
ax.plot(log_xs,mean_prob_m,color=m_color,label='Size matched')
ax.plot(log_xs,np.clip(mean_prob_m - 2*std_prob_m,0,1),linestyle=':',color=m_color)
ax.plot(log_xs,np.clip(mean_prob_m + 2*std_prob_m,0,1),linestyle=':',color=m_color)

ax.plot(log_xs,mean_prob_s,color=s_color,label='Vs. smaller',linestyle='dashed')
ax.plot(log_xs,np.clip(mean_prob_s - 2*std_prob_s,0,1),linestyle=':',color=s_color)
ax.plot(log_xs,np.clip(mean_prob_s + 2*std_prob_s,0,1),linestyle=':',color=s_color)

ax.plot(log_xs,mean_prob_b,color=b_color,label='Vs. bigger',linestyle='dashdot')
ax.plot(log_xs,np.clip(mean_prob_b - 2*std_prob_b,0,1),linestyle=':',color=b_color)
ax.plot(log_xs,np.clip(mean_prob_b + 2*std_prob_b,0,1),linestyle=':',color=b_color)

ax.set_xticks(log_xs)
ax.set_xticklabels(xs)
ax.set_xlabel('n Replicates')
ax.set_ylabel('P(getting significant result)')
ax.set_ylim([-0.1,1.1])

#ax.fill_between(xs,mean_prob_m - std_prob_m,mean_prob_m + std_prob_m)

## Plot the mean effect strength (with avg SEM for r replicates)
fig1,ax1 = plt.subplots()
#print(log_xs,mean_diff_m)
#print(mean_diff_s,mean_diff_b)
ax1.plot(log_xs,mean_diff_m,color=m_color,label='Size matched')
ax1.plot(log_xs,mean_diff_m - sem_diff_m,linestyle=':',color=m_color)
ax1.plot(log_xs,mean_diff_m + sem_diff_m,linestyle=':',color=m_color)

ax1.plot(log_xs,mean_diff_s,color=s_color,label='vs. smaller')
ax1.plot(log_xs,mean_diff_s - sem_diff_s,linestyle=':',color=s_color)
ax1.plot(log_xs,mean_diff_s + sem_diff_s,linestyle=':',color=s_color)

ax1.plot(log_xs,mean_diff_b,color=b_color,label='vs. bigger')
ax1.plot(log_xs,mean_diff_b - sem_diff_b,linestyle=':',color=b_color)
ax1.plot(log_xs,mean_diff_b + sem_diff_b,linestyle=':',color=b_color)

ax1.axhline(0,color='grey')
ax1.set_xticks(log_xs)
ax1.set_xticklabels(xs)
ax1.set_xlabel('n Replicates')
ax1.set_ylabel('Mean difference observed (winner vs loser)')

ax.legend()

if False:
    fig1.savefig('./figures/fig4S_diff.svg')
    fig.savefig('./figures/fig4S_prob.svg')
if True:
    plt.show()

if np.nanmax(mean_diff_b) > 1:
    import pdb;pdb.set_trace()
if np.nanmax(mean_diff_s) > 1:
    import pdb;pdb.set_trace()
if np.nanmax(mean_diff_m) > 1:
    import pdb;pdb.set_trace()



print('Done!')

