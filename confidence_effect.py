#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish
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

params_bayes = Params()
params_bayes.effort_method = [1,1]
params_bayes.n_fights = 5
params_bayes.n_fish = 5
params_bayes.f_method = 'balanced'
params_bayes.f_outcome = 'math'
params_bayes.outcome_params = [s,e,l]
params_bayes.set_L()
params_bayes.u_method = 'bayes'

params_boost = copy.deepcopy(params_bayes)
params_boost.u_method = 'size_boost'

NAIVE = True

def run_tanks(naive=True,params=Params()):
    drop_count = 0
    winners,losers = [],[]
    PLOT = 'estimate'

## Let fish duke it out, then pull a fish out, check it's success against a size matched fish, and put it back in
    iterations = 500
    scale = 1
    f0 = Fish(0,params)
    results = [[],[]]
    if naive:
        pre_rounds = 0
    else:
        print('doing pre rounds this time')
        pre_rounds = 5
    params.n_fights = pre_rounds
    for i in tqdm(range(iterations)):
#for i in range(iterations):
        fishes = [Fish(f,params,likelihood=f0.naive_likelihood) for f in range(params.n_fish)]
        tank = Tank(fishes,params)
        tank._initialize_likelihood() ## This used to be super important, without intiializing it, it takes ages. 
        tank.run_all(progress=False)

        focal_fish = fishes[0]
        if focal_fish.estimate < 8 or focal_fish.estimate > 99:
            drop_count += 1
            #pass
            continue
        focal_fish.i_shift = len(focal_fish.est_record) - 1
        #matched_fish = Fish(0,size=focal_fish.size,prior=True,effort_method=params.effort_method,update_method=params.u_method)
        matched_fish = copy.deepcopy(focal_fish)
        matched_fish.idx = 10
        match = Fight(matched_fish,focal_fish,params,outcome=np.random.randint(2))
        outcome = match.run_outcome()
        focal_fish.update(outcome,match)


        match2 = Fight(matched_fish,focal_fish,params)
        test_outcome = match2.run_outcome()
        if outcome == 1:
            winners.append(focal_fish)
        else:
            losers.append(focal_fish)
        results[outcome].append(test_outcome)

        tank2 = Tank(fishes,params)
        tank2.run_all(False)
    print('dropped n crappy fish:',drop_count)
    return winners,losers,results


def plot_tanks(winners,losers,naive=True,ax=None,shift=0):
    if ax is None:
        fig,ax = plt.subplots()

    i_shift = winners[0].i_shift
    loser_array = np.empty([len(losers),len(losers[0].est_record)])
    winner_array = np.empty([len(winners),len(winners[0].est_record)])

    for f_i in range(len(losers)):
        f = losers[f_i]
        #ax.plot(f.est_record - f.est_record[i_shift],color='darkblue',alpha=.01)
        loser_array[f_i] = f.est_record - f.est_record[i_shift]
    for f_i in range(len(winners)):
        f = winners[f_i]
        #ax.plot(f.est_record - f.est_record[i_shift],color='gold',alpha=.01)
        winner_array[f_i] = f.est_record - f.est_record[i_shift]

    mean_win = np.mean(winner_array,0)
    sem_win = np.std(winner_array,0) / np.sqrt(len(winner_array))

    mean_lose = np.mean(loser_array,0)
    sem_lose = np.std(loser_array,0) / np.sqrt(len(loser_array))

    mean_win += shift
    mean_lose += shift
#ax.plot(mean_win,color='gold',linewidth=5)
    xs = np.arange(len(mean_win)) - i_shift
    if naive:
        alph = .4
        ls = 'solid'
        win_color = 'lemonchiffon'
        lose_color = 'royalblue'
        if shift != 0:
            label = 'naive'
        else:
            label = None
    else:
        alph = .8
        ls = 'dashed'
        win_color = 'gold'
        lose_color = 'darkblue'
        if shift != 0:
            label = 'experienced'
        else:
            label = None
    ax.plot(xs,mean_win,color='black',linestyle=ls,label=label)
    ax.fill_between(xs,mean_win+sem_win,mean_win-sem_win,color=win_color,alpha=alph)

#ax.plot(np.mean(loser_array,0),color='darkblue',linewidth=5)
    ax.plot(xs,mean_lose,color='lightgray',linestyle=ls)
    ax.fill_between(xs,mean_lose+sem_lose,mean_lose-sem_lose,color=lose_color,alpha=alph)

    ax.set_xlabel('Fights since size-matched challenge')
    ax.set_ylabel('Difference in estimate')
    return ax

#fig2,ax2 = plt.subplots()

winners_naive,losers_naive,results_naive = run_tanks(naive=True,params=params_bayes)
winners_exp,losers_exp,results_exp = run_tanks(naive=False,params=params_bayes)

if False:
    fig,ax = plt.subplots()
    ax = plot_tanks(winners_naive,losers_naive,True,ax)
    ax = plot_tanks(winners_exp,losers_exp,False,ax)
print('#### Bayes updating: ####')
print('naive winner win-rate:',np.mean(results_naive[1]),np.std(results_naive[1] / np.sqrt(len(results_naive[1]))))
print('naive loser win-rate:',np.mean(results_naive[0]),np.std(results_naive[0] / np.sqrt(len(results_naive[0]))))

print('exp winner win-rate:',np.mean(results_exp[1]),np.std(results_exp[1] / np.sqrt(len(results_exp[1]))))
print('exp loser win-rate:',np.mean(results_exp[0]),np.std(results_exp[0] / np.sqrt(len(results_exp[0]))))

print('exp winner estimate:',np.mean([f.estimate for f in winners_exp]))
print('exp winner size:',np.mean([f.size for f in winners_exp]))

fig2,ax2 = plt.subplots()
## Bars are Winners (naive vs exp) and Losers (naive vs exp)

bars = [np.mean(results_naive[1]),np.mean(results_exp[1]),np.mean(results_naive[0]),np.mean(results_exp[0])]
err = np.array([np.std(results_naive[1])/ np.sqrt(len(results_naive[1])),
                np.std(results_exp[1])/ np.sqrt(len(results_exp[1])),
                np.std(results_naive[0])/ np.sqrt(len(results_naive[0])),
                np.std(results_exp[0])/ np.sqrt(len(results_exp[0]))])

bars = np.array(bars) - .5

ax2.bar([1,0,1,0],bars,bottom=.5,yerr=err,color=['#FFD70040','gold','#00008B40','darkblue'])
ax2.axhline(.5,linestyle=':',color='black')
ax2.set_ylim([0.2,0.8])
#fig2.show()
fig2.savefig('./imgs/showme.png')
plt.show()

b_shift = -15
winners_naive,losers_naive,results_naive = run_tanks(naive=True,params=params_boost)
winners_exp,losers_exp,results_exp = run_tanks(naive=False,params=params_boost)
if False:
    ax = plot_tanks(winners_naive,losers_naive,True,ax,shift = b_shift)
    ax = plot_tanks(winners_exp,losers_exp,False,ax,shift = b_shift)

print('#### Simple Boost updating: ####')
print('naive winner win-rate:',np.mean(results_naive[1]),np.std(results_naive[1] / np.sqrt(len(results_naive[1]))))
print('naive loser win-rate:',np.mean(results_naive[0]),np.std(results_naive[0] / np.sqrt(len(results_naive[0]))))

print('exp winner win-rate:',np.mean(results_exp[1]),np.std(results_exp[1] / np.sqrt(len(results_exp[1]))))
print('exp loser win-rate:',np.mean(results_exp[0]),np.std(results_exp[0] / np.sqrt(len(results_exp[0]))))

if False:
    ax.axhline(-7.5,linewidth=5,color='black')
    ax.axvline(0,color='black',linestyle=':')
    ax.set_xlim([-0.5,2.1])
    ax.set_ylim([-22,7])

    ax.set_yticks(np.arange(-20,7,5))
    ax.set_yticklabels([-5,0,5,-5,0,5])
#ax.set_xticks(np.arange(-1,3 + 1,2))
#ax.set_xticklabels([-2,0,2,4,6,8,10])

    fig.legend(loc='upper left')
#fig2.show()
#fig.show()

#plt.show()

