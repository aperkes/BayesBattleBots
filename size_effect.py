#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

from tqdm import tqdm

import random, copy

s,e,l = .6,.3,.01

params_bayes = SimParams()
params_bayes.effort_method = [1,1]
params_bayes.n_fights = 5
params_bayes.n_fish = 6
params_bayes.f_method = 'balanced'
params_bayes.f_outcome = 'math'
params_bayes.outcome_params = [s,e,l]
params_bayes.u_method = 'bayes'

params_boost = copy.deepcopy(params_bayes)
params_boost.u_method = 'size_boost'

#s = Simulation(params)

NAIVE = True

def run_tanks(naive=False,params=SimParams()):
    drop_count = 0
    equals,littles = [],[]
    equal_losers,little_losers = [],[]

    results = [[],[]]
    loss_results = [[],[]]
## Let fish duke it out, then pull a fish out, check it's success against a size matched fish, and put it back in
    iterations = 1000
    f0 = Fish(0,effort_method=params.effort_method,update_method=params.u_method)
    print('u_method',params.u_method)
    if naive:
        pre_rounds = 0
    else:
        print('doing pre rounds this time')
        pre_rounds = 10
    for i in tqdm(range(iterations)):
    #for i in range(iterations):
        fishes = [Fish(f,likelihood=f0.naive_likelihood,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]
        tank = Tank(fishes,n_fights = pre_rounds,f_params=params.outcome_params,f_method=params.f_method,f_outcome=params.f_outcome,u_method=params.u_method)
        tank._initialize_likelihood() ## This is super important, without intiializing it, it takes ages. 
        tank.run_all(False)

        focal_fish = fishes[0]
        focal_fish.i_shift = len(focal_fish.est_record) - 1 ## This index of the estimate just before the forced fight

        focal_fish2 = copy.deepcopy(fishes[0])

        if focal_fish.estimate < 8 or focal_fish.estimate > 99:
            drop_count += 1
            continue

        pre_est = focal_fish.estimate

        if True: #Sanity check....The IDX MATTERS!!!!
            matched_fish = copy.deepcopy(focal_fish) 
            matched_fish2 = copy.deepcopy(focal_fish) 
            loser_match =  Fish(10,size=focal_fish.size,prior=True,effort_method=params.effort_method,update_method=params.u_method)
            loser_small =  Fish(10,size=20,prior=True,effort_method=params.effort_method,update_method=params.u_method)

            match_win =  Fight(focal_fish,loser_match,outcome_params=params.outcome_params,outcome=0)
            small_win =  Fight(focal_fish2,loser_small,outcome_params=params.outcome_params,outcome=0)

            focal_loser = copy.deepcopy(focal_fish)
            focal_loser_small = copy.deepcopy(focal_fish)

            match_loss =  Fight(focal_fish,loser_match,outcome_params=params.outcome_params,outcome=1)
            small_loss =  Fight(focal_fish2,loser_small,outcome_params=params.outcome_params,outcome=1)


            match_loss.run_outcome()
            small_loss.run_outcome()

            match_win.run_outcome()
            small_win.run_outcome()

            focal_fish.update(True,match_win)
            focal_fish2.update(True,small_win)

            focal_loser.update(False,match_loss)
            focal_loser_small.update(False,small_loss)

            #print('results so far:',focal_fish.win_record[-1],focal_fish2.win_record[-1])
            #print(focal_fish.est_record[-2:],focal_fish2.est_record[-2:])
            if False:
                assay_fight = Fight(focal_fish,matched_fish,outcome_params=params.outcome_params)
                assay_fight2 = Fight(focal_fish2,matched_fish2,outcome_params=params.outcome_params)
            else:
                assay_fight = Fight(matched_fish,focal_fish,outcome_params=params.outcome_params)
                assay_fight2 = Fight(matched_fish2,focal_fish2,outcome_params=params.outcome_params)
                assay_fight3 = Fight(matched_fish,focal_loser,outcome_params=params.outcome_params)
                assay_fight4 = Fight(matched_fish,focal_loser_small,outcome_params=params.outcome_params)
            #print(focal_fish.estimate,focal_fish2.estimate,matched_fish.estimate)
            #print(focal_fish.estimate,matched_fish.estimate)
            assay_outcome = assay_fight.run_outcome()
            if params.u_method == 'size_boost':
                pass
                #print(focal_fish.effort,matched_fish.effort)
                #print(matched_fish.effort)

            assay_outcome2 = assay_fight2.run_outcome()
            assay_outcome3 = assay_fight3.run_outcome()
            assay_outcome4 = assay_fight4.run_outcome()

            #print(matched_fish.effort)

            littles.append(focal_fish2)
            equals.append(focal_fish)

            equal_losers.append(focal_loser)
            little_losers.append(focal_loser_small)
            #print('effort:',focal_fish.effort,focal_fish2.effort,matched_fish.effort)

            results[1].append(assay_outcome2)
            results[0].append(assay_outcome)

            loss_results[1].append(assay_outcome4)
            loss_results[0].append(assay_outcome3)

        else:
            if np.random.rand() > .5:
                fish_size = focal_fish.size
                little = False
            else:
                fish_size = 20
                little = True
            loser_fish = Fish(0,size=fish_size,prior=True,effort_method=params.effort_method,update_method=params.u_method)

            #matched_fish = Fish(0,size=focal_fish.size,prior=True,effort_method=params.effort_method,update_method=params.u_method)
            matched_fish = copy.deepcopy(focal_fish)
            matched_fish.idx = 1

            winning_fight = Fight(focal_fish,loser_fish,outcome_params=params.outcome_params,outcome=0)
            outcome = winning_fight.run_outcome()
            #print(winning_fight.winner.idx,outcome)
            #print(focal_fish.estimate)
            focal_fish.update(1-outcome,winning_fight)
            #print(focal_fish.win_record[-1])
            #print(pre_est,focal_fish.estimate)
            #print(focal_fish.estimate)

            match_fight = Fight(focal_fish,matched_fish,outcome_params=params.outcome_params)
            test_outcome = match_fight.run_outcome()
            if little:
                littles.append(focal_fish)
                results[1].append(1-test_outcome)
            else:
                equals.append(focal_fish) 
                results[0].append(1-test_outcome)

    print('dropped n crappy fish:',drop_count)
    return [equals,littles,results],[equal_losers,little_losers,loss_results]

#fig2,ax2 = plt.subplots()

[equals_bayes,littles_bayes,results_bayes],[equals_bayes_loss,littles_bayes_loss,results_bayes_loss] = run_tanks(naive=False,params=params_bayes)
[equals_boost,littles_boost,results_boost],[equals_boost_loss,littles_boost_loss,results_boost_loss] = run_tanks(naive=False,params=params_boost)
#ax = plot_tanks(winners_bayes,naive=False,ax=ax)

print('#### Bayes updating: ####')
bayes_mean_equal = np.mean(results_bayes[0])
bayes_sem_equal = np.std(results_bayes[0]) / np.sqrt(len(results_bayes[0]))

bayes_mean_little = np.mean(results_bayes[1])
bayes_sem_little = np.std(results_bayes[1]) / np.sqrt(len(results_bayes[1]))

bayes_mean_equal_loss = np.mean(results_bayes_loss[0])
bayes_sem_equal_loss = np.std(results_bayes_loss[0]) / np.sqrt(len(results_bayes_loss[0]))

bayes_mean_little_loss = np.mean(results_bayes_loss[1])
bayes_sem_little_loss = np.std(results_bayes_loss[1]) / np.sqrt(len(results_bayes_loss[1]))
print('equal win-rate:',bayes_mean_equal,bayes_sem_equal)
print('little win-rate:',bayes_mean_little,bayes_sem_little)
print('equal win-rate (after loss):',bayes_mean_equal_loss,bayes_sem_equal_loss)
print('little win-rate (after loss):',bayes_mean_little_loss,bayes_sem_little_loss)
#print('equal win-rate:',np.mean(results_bayes[0]),np.std(results_bayes[0] / np.sqrt(len(results_bayes[0]))))
#print('little win-rate:',np.mean(results_bayes[1]),np.std(results_bayes[1] / np.sqrt(len(results_bayes[1]))))


print('#### Boost updating: ####')
boost_mean_equal = np.mean(results_boost[0])
boost_sem_equal = np.std(results_boost[0]) / np.sqrt(len(results_boost[0]))

boost_mean_little = np.mean(results_boost[1])
boost_sem_little = np.std(results_boost[1]) / np.sqrt(len(results_boost[1]))

boost_mean_equal_loss = np.mean(results_boost_loss[0])
boost_sem_equal_loss = np.std(results_boost_loss[0]) / np.sqrt(len(results_boost_loss[0]))

boost_mean_little_loss = np.mean(results_boost_loss[1])
boost_sem_little_loss = np.std(results_boost_loss[1]) / np.sqrt(len(results_boost_loss[1]))


print('equal win-rate:',boost_mean_equal,boost_sem_equal)
print('little win-rate:',boost_mean_little,boost_sem_little)

print('equal win-rate (post loss):',boost_mean_equal_loss,boost_sem_equal_loss)
print('little win-rate (post loss):',boost_mean_little_loss,boost_sem_little_loss)
#print('equal win-rate:',np.mean(results_bayes[0]),np.std(results_bayes[0] / np.sqrt(len(results_bayes[0]))))
#print('little win-rate:',np.mean(results_bayes[1]),np.std(results_bayes[1] / np.sqrt(len(results_bayes[1]))))
## Look at size diff vs winer effect


if True:
    equal_shift = [f.est_record[f.i_shift+1] - f.est_record[f.i_shift] for f in equals_bayes]
    little_shift = [f.est_record[f.i_shift+1] - f.est_record[f.i_shift] for f in littles_bayes]

    equal_shift_boost = [f.est_record[f.i_shift+1] - f.est_record[f.i_shift] for f in equals_boost]
    little_shift_boost = [f.est_record[f.i_shift+1] - f.est_record[f.i_shift] for f in littles_boost]
    #equal_shift_ = [f.est_record_[f.i_shift+1] - f.est_record_[f.i_shift] for f in equals_bayes]
    #little_shift_ = [f.est_record_[f.i_shift+1] - f.est_record_[f.i_shift] for f in littles_bayes]

    print(np.mean(equal_shift),np.mean(little_shift))
    print(np.std(equal_shift),np.std(little_shift))

    print(np.mean(equal_shift_boost),np.mean(little_shift_boost))
    print(np.std(equal_shift_boost),np.std(little_shift_boost))
    #print(np.mean(equal_shift_),np.mean(little_shift_))
    #print(np.std(equal_shift_),np.std(little_shift_))

bars = [bayes_mean_equal,bayes_mean_little,boost_mean_equal,boost_mean_little]
errs = [bayes_sem_equal,bayes_sem_little,boost_sem_equal,boost_sem_little]

loser_bars = [bayes_mean_equal_loss,bayes_mean_little_loss,boost_mean_equal_loss,boost_mean_little_loss]
loser_errs = [bayes_sem_equal_loss,bayes_sem_little_loss,boost_sem_equal_loss,boost_sem_little_loss]

bars = np.array(bars) - .5
loser_bars = np.array(loser_bars) - .5

fig,ax = plt.subplots()

ax.bar([0,1,3,4],bars,bottom=.5,yerr=errs,color=['gold','#FFD70040','gold','#FFD70040'])
ax.bar([0,1,3,4],loser_bars,bottom=.5,yerr=errs,color=['darkblue','#00008B40','darkblue','#00008B40'])

ax.axhline(.5,color='black',linestyle=':')
ax.set_ylim([0.2,0.8])

fig.savefig('./imgs/size_effect.svg',dpi=300)
plt.show()

## I don't need this, but it's here if I change my mind:
def plot_tanks(winners,naive=True,ax=None,shift=0):
    if ax is None:
        fig,ax = plt.subplots()

    i_shift = winners[0].i_shift
    winner_array = np.empty([len(winners),len(winners[0].est_record)])

    for f_i in range(len(winners)):
        f = winners[f_i]
        #ax.plot(f.est_record - f.est_record[i_shift],color='gold',alpha=.01)
        winner_array[f_i] = f.est_record - f.est_record[i_shift]

    mean_win = np.mean(winner_array,0)
    sem_win = np.std(winner_array,0) / np.sqrt(len(winner_array))

    mean_win += shift
#ax.plot(mean_win,color='gold',linewidth=5)
    xs = np.arange(len(mean_win)) - i_shift
    if naive:
        alph = .4
        ls = 'solid'
        if shift != 0:
            label = 'naive'
        else:
            label = None
    else:
        alph = .8
        ls = 'dashed'
        if shift != 0:
            label = 'experienced'
        else:
            label = None
    ax.plot(xs,mean_win,color='black',linestyle=ls,label=label)
    ax.fill_between(xs,mean_win+sem_win,mean_win-sem_win,color='gold',alpha=alph)


    ax.set_xlabel('Fights since size-matched challenge')
    ax.set_ylabel('Difference in estimate')
    return ax


