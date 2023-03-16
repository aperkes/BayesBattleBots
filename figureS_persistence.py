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

#s,e,l = -.8,-.6,-0.99
s,e,l=-0.9,-0.5,-0.7

params = Params()
params.mean_size = 50
params.sd_size = 20
#params.poly_param_c = 0
#params.effort_method = [1,1]
params.n_fights = 5 
params.n_fish = 5
params.acuity = 5
params.f_method = 'balanced'
params.u_method = 'bayes'
params.f_outcome = 'math'
params.outcome_params = [s,e,l]
params.set_params()
## Let fish duke it out, then pull a fish out, let it win, and put it back in with the rest.
replicates = 10
iterations = 3
scale = 1


#plt.rcParams.update({'font.size': 18})
#plt.rcParams.update({'lines.linewidth':5})
#plt.rcParams.update({'axes.linewidth':2})

INIT = False

ew_pairs = []

min_reps = 10
max_reps = 20
r_bins = max_reps - min_reps

ps_array = np.empty([r_bins,iterations])
pb_array = np.empty_like(ps_array)
pm_array = np.empty_like(ps_array)
smean_array = np.empty_like(ps_array)
bmean_array = np.empty_like(ps_array)
mmean_array = np.empty_like(ps_array)

for reps in tqdm(range(min_reps,max_reps)):
    for i in range(iterations): 
        winners,losers = [],[]
        biggers,smallers = [],[]
        winner_v_bigger,loser_v_bigger = [],[]
        winner_v_smaller,loser_v_smaller = [],[]
        winner_record,loser_record = [],[]
        final_win,final_loss = [],[]

        for r in range(replicates):
            fishes = [Fish(f,params) for f in range(params.n_fish)]
            tank = Tank(fishes,params)
            if INIT:
                tank._initialize_likelihood()
            tank.run_all(False)

            f = copy.deepcopy(tank.fishes[0])
            f2 = copy.deepcopy(tank.fishes[1])
            opp = Fish(size = f.size*scale)
            opp2 = Fish(size = f2.size/scale)
            f_win = Fight(f,opp,outcome=0)
            f_loss = Fight(f2,opp2,outcome=1) 
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
            #print(f.estimate)
            fishes2 = copy.deepcopy(fishes)
            fishes2[0] = f
            tank2 = Tank(fishes2,params)
            if INIT:
                tank2._initialize_likelihood()
            fishes3 = copy.deepcopy(fishes)
            fishes3[1] = f2
            tank3 = Tank(fishes3,params)
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

            winner_record.append(np.sum(tank2.history[-1,f.idx]) / (params.n_fish - 1))
            loser_record.append(np.sum(tank3.history[-1,f2.idx]) / (params.n_fish - 1))

            if np.max(tank2.sizes) != f2.size:
                idx_bigger_l = np.arange(tank.n_fish)[ranks == (l_rank + 1)][0]
                loser_v_bigger.append(tank3.win_record[f2.idx,idx_bigger_l]/params.n_fights)
                #loser_v_bigger.append(tank3.history[-1,f2.idx,idx_bigger_l])

            if np.min(tank2.sizes) != f.size:
                idx_smaller_w = np.arange(tank.n_fish)[ranks == (f_rank - 1)][0]
                winner_v_smaller.append(tank2.win_record[f.idx,idx_smaller_w]/params.n_fights)
                #winner_v_smaller.append(tank2.history[-1,f.idx,idx_smaller_w])

            if np.max(tank2.sizes) != f.size:
                idx_bigger = np.arange(tank.n_fish)[ranks == (f_rank + 1)][0]
                winner_v_bigger.append(tank2.win_record[f.idx,idx_bigger]/params.n_fights)
                #winner_v_bigger.append(tank2.history[-1,f.idx,idx_bigger])

            else:
                idx_bigger = f.idx 
            biggers.append(tank.fishes[idx_bigger])

            if np.min(tank3.sizes) != f2.size:
                idx_smaller = np.arange(tank.n_fish)[ranks == (l_rank - 1)][0]
                loser_v_smaller.append(tank3.win_record[f2.idx,idx_smaller]/params.n_fights)
                #loser_v_smaller.append(tank3.history[-1,f2.idx,idx_smaller])
                #import pdb;pdb.set_trace()
            else:
                idx_smaller = f2.idx
            smallers.append(tank2.fishes[idx_smaller])
            losers.append(f2)

            match1 = Fish(1,params,size = f.size)
            match2 = Fish(2,params,size = f2.size)
            #print(f.estimate,f2.estimate,match1.estimate,match2.estimate)
            fight1 = Fight(f,match1,params)
            fight2 = Fight(f2,match2,params)
            fight1.run_outcome()
            ew_pairs.append([f.effort,f.wager])
            #import pdb;pdb.set_trace()
            fight2.run_outcome()
            #print(f2.effort,match2.effort,f.effort)
            final_win.append(1-fight1.outcome)
            final_loss.append(1-fight2.outcome)
         
        win_outcomes = []
        loss_outcomes = []


        for f_i in range(len(winners)):
            f = winners[f_i]
            check_fish = Fish(0,params,size=f.size)
            check_fight = Fight(check_fish,f,params)
            outcome = check_fight.run_outcome()
            win_outcomes.append(outcome)

        for f_i in range(len(losers)):
            f = losers[f_i]

### Check loser against a size matched fish
            check_fish = Fish(0,params,size=f.size)
            check_fight = Fight(check_fish,f,params)
            outcome = check_fight.run_outcome()
            loss_outcomes.append(outcome)


           
        t_s,p_s = ttest_ind(winner_v_smaller,loser_v_smaller)
        t_b,p_b = ttest_ind(winner_v_bigger,loser_v_bigger)
        t_m,p_m = ttest_ind(win_outcomes,loss_outcomes)
        s_diff = np.mean(winner_v_smaller) - np.mean(loser_v_smaller)
        b_diff = np.mean(winner_v_bigger) - np.mean(loser_v_bigger)
        m_diff = np.mean(win_outcomes) - np.mean(loss_outcomes)
        ps_array[r,i] = p_s
        pb_array[r,i] = p_b
        pm_array[r,i] = p_b
        smean_array[r,i] = s_diff
        bmean_array[r,i] = b_diff
        mmean_array[r,i] = m_diff

fig,ax = plt.subplots()
ax.plot(np.arange(min_reps,max_reps),np.nanmean(pm_array,1))
plt.show()
"""
if False:
    fig5,ax5 = plt.subplots()
    ew_pairs = np.array(ew_pairs)
    ax5.scatter(ew_pairs[:,0],ew_pairs[:,1])

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

winner_array = np.empty([len(winners),len(winners[0].est_record)])
loser_array = np.empty([len(losers),len(losers[0].est_record)])


win_outcomes = []
loss_outcomes = []


for f_i in range(len(winners)):
    f = winners[f_i]
    check_fish = Fish(0,params,size=f.size)
    check_fight = Fight(check_fish,f,params)
    outcome = check_fight.run_outcome()
    win_outcomes.append(outcome)

    pre_success = np.mean(np.array(f.win_record)[:n_rounds-1,1])
    post_success = np.mean(np.array(f.win_record)[n_rounds:,1])
    pre_estimate = np.mean(np.array(f.est_record)[:n_rounds-1])
    post_estimate = np.mean(np.array(f.est_record)[n_rounds:])
    win_pre.append(pre_success)
    win_post.append(post_success)
    est_pre.append(pre_estimate)
    est_post.append(post_estimate)
    #f = tank.fishes[0]
    winner_array[f_i] = f.est_record-f.est_record[n_rounds-1]
    ax.plot(np.array(f.est_record)-f.est_record[n_rounds-1],color='gold',alpha=.01)
    jitter = (np.random.rand() - .5) * .01
    #ax.plot(np.array(f.win_record)[:,1] + jitter,color='green',alpha=.01)
    if biggers[f_i].idx != f.idx:
        est_diff = biggers[f_i].size-f.est_record[n_rounds-1]
        est_bigs.append(est_diff) ## Careful...
        #ax.axhline(biggers[f_i].size-f.est_record[n_rounds-1],color='red',alpha=.1)
        #ax.axhline(biggers[f_i].est_record[n_rounds + 1]-f.est_record[n_rounds-1],color='blue',alpha=.1)
for f_i in range(len(losers)):
    f = losers[f_i]

### Check loser against a size matched fish
    check_fish = Fish(0,params,size=f.size)
    check_fight = Fight(check_fish,f,params)
    outcome = check_fight.run_outcome()
    loss_outcomes.append(outcome)

    pre_success = np.mean(np.array(f.win_record)[:n_rounds-1,1])
    post_success = np.mean(np.array(f.win_record)[n_rounds:,1])
    loser_pre.append(pre_success)
    loser_post.append(post_success)
    ys = np.array(f.est_record) - f.est_record[n_rounds-1]
    loser_array[f_i] = ys

    ax.plot(ys,color='darkblue',alpha=.01)
    if smallers[f_i].idx != f.idx:
        est_diff = smallers[f_i].size-f.est_record[n_rounds-1]
        est_smalls.append(est_diff) ## Careful...
    #print(f.est_record)

winner_mean = np.nanmean(winner_array,0)
winner_sem = np.nanstd(winner_array,0) / np.sqrt(len(winner_array))

loser_mean = np.nanmean(loser_array,0)
loser_sem = np.nanstd(loser_array,0) / np.sqrt(len(loser_array))

ax.plot(winner_mean,color='gold')
ax.fill_between(np.arange(len(winner_mean)),winner_mean+winner_sem,winner_mean-winner_sem,alpha=.8,color='gold')

ax.plot(loser_mean,color='darkblue')
ax.fill_between(np.arange(len(loser_mean)),loser_mean+loser_sem,loser_mean-loser_sem,alpha=.8,color='darkblue')
if True: ## Show the size to next biggest and next smallest
    ax.axhline(np.mean(est_bigs),color='grey',linestyle=':')
    ax.axhline(np.mean(est_smalls),color='grey',linestyle=':',alpha=0.5)
    ax.set_ylim([np.mean(est_smalls)-2,np.mean(est_bigs)+2])
else:
    ax.set_ylim([np.mean(est_smalls)-1,np.mean(est_bigs)+1])

ax.axvline(params.n_fights * (params.n_fish-1),color='black',linestyle=':',label='forced win/loss')
ax.axhline(0,color='black',label='estimate prior to staged fight')
ax.set_xlim([n_rounds-6.5,n_rounds+14.5])
ax.set_xticks(np.arange(15,37,5))
ax.legend(loc='upper right')
ax.set_ylabel('Normalized size estimate')
ax.set_xlabel('Contest number')
ax.set_title('The winner effect persists indefinitely')

fig.tight_layout()
#fig.savefig('./imgs/win_loss.png',dpi=300)
#fig.savefig('./imgs/win_loss.svg')
# Calculate probability as a function of steps: 
win_array = np.array([f.win_record for f in winners])[:,:,1]
loser_array =np.array([f.win_record for f in losers])[:,:,1]

avg_winners = np.mean(win_array,axis=0)
avg_losers = np.mean(loser_array,axis=0)

print('before after win:')
print(np.nanmean(win_pre),np.nanmean(win_post))
print(np.nanmean(est_pre),np.nanmean(est_post))
print('before after loss:')
print(np.mean(loser_pre),np.mean(loser_post))

if False:
    fig1,ax1 = plt.subplots()

    ax1.plot(avg_winners,color='gold')
    ax1.plot(avg_losers,color='darkblue')
    ax1.axhline(.5,color='black',linestyle=':')

    ax1.set_xlabel('Contest number')
    ax1.set_ylabel('Win Proportion')
    ax1.set_title('Wins/Losses may not be visible in social groups')
    fig1.show()

#print('pre post win rate:')
#print(win_pre,win_post)
#print(loser_pre,loser_post)


fig2,ax2 = plt.subplots()

ax2.bar(0,np.mean(win_outcomes)-0.5,bottom=0.5,color='gold',yerr=np.std(win_outcomes)/np.sqrt(len(winners)))
ax2.bar(1,np.mean(loss_outcomes)-0.5,bottom=0.5,color='darkblue',yerr=np.std(loss_outcomes)/np.sqrt(len(winners)))
ax2.axhline(0.5,linestyle=':',color='black')
ax2.set_ylim([0.0,1.0])
ax2.set_ylabel('Proportion of fights won')

ax2.set_xticks([0,1])
ax2.set_xticklabels(['Winners','Losers'],rotation=45)
ax2.set_title('The winner effect is detectable in size-matched fights')

fig2.tight_layout()

fig3,ax3 = plt.subplots()

if False:
    ax3.bar(0,np.mean(winner_v_bigger),color='gold',yerr=np.std(winner_v_bigger)/np.sqrt(len(winners)))
    ax3.bar(1,np.mean(loser_v_bigger),color='darkblue',yerr=np.std(loser_v_bigger)/np.sqrt(len(losers)))
else:
    ax3.bar(0,np.mean(winner_v_bigger)-0.5,bottom=0.5,color='gold',yerr=np.std(winner_v_bigger)/np.sqrt(len(winners)))
    ax3.bar(1,np.mean(loser_v_bigger)-0.5,bottom=0.5,color='darkblue',yerr=np.std(loser_v_bigger)/np.sqrt(len(losers)))
    ax3.axhline(0.5,linestyle=':',color='black')
ax3.set_ylim([0.0,1.0])
ax3.set_ylabel('Proportion of fights won')

fig3.tight_layout()

fig4,ax4 = plt.subplots()

if False:
    ax4.bar(0,np.mean(winner_v_smaller),color='gold',yerr=np.std(winner_v_smaller)/np.sqrt(len(winners)))
    ax4.bar(1,np.mean(loser_v_smaller),color='darkblue',yerr=np.std(loser_v_smaller)/np.sqrt(len(losers)))
else:
    ax4.bar(0,np.mean(winner_v_smaller)-0.5,bottom=0.5,color='gold',yerr=np.std(winner_v_smaller)/np.sqrt(len(winners)))
    ax4.bar(1,np.mean(loser_v_smaller)-0.5,bottom=0.5,color='darkblue',yerr=np.std(loser_v_smaller)/np.sqrt(len(losers)))
    ax4.axhline(0.5,linestyle=':',color='black')
ax4.set_ylim([0.0,1.0])
ax4.set_ylabel('Proportion of fights won')

fig4.tight_layout()

#import pdb;pdb.set_trace()
print('mean winner,loser')
print(np.mean(winner_record),np.mean(loser_record))
plt.show()
print(f.params.outcome_params)
"""
print('Done!')

