
import numpy as np
from matplotlib import pyplot as plt

from fish import Fish,FishNPC
from fight import Fight
from params import Params

from tqdm import tqdm

## A couple of helper functions to keep things tidy
def mean_sem(a):
    mean_a = np.mean(a)
    sem_a = np.std(a) / np.sqrt(len(a))
    return mean_a,sem_a

def plot_fill(xs,a,fig=None,ax=None,color='grey',alpha=0.5):
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(xs,a[:,0],color='black')
    ax.fill_between(xs,a[:,0] - a[:,1],a[:,0] + a[:,1],color=color,alpha=alpha)
    return fig,ax

def run_sim(params):
    iterations = params.iterations

    outcome_array = np.empty([iterations,2])
    outcome_array.fill(np.nan)

    win_info_array = np.array(outcome_array)
    loss_info_array = np.array(outcome_array)

    for i in tqdm(range(iterations)):
        focal_winner = Fish(i+2,params)
        focal_loser = focal_winner.copy() 

        f_sizes.append(focal_winner.size)
## Stage a bunch of wins and losses against size-matched fish

        staged_opp = Fish(0,params)
        
        staged_win = Fight(staged_opp,focal_winner,params,outcome=1)
        staged_win.run_outcome()
        focal_winner.update(True,staged_win)

        staged_loss = Fight(staged_opp,focal_loser,params,outcome=0)
        staged_loss.run_outcome()
        focal_loser.update(False,staged_loss)

## Assay against size matched fish
        assay_fish = Fish(1,assay_params)

        assay_winner = Fight(assay_fish,focal_winner,params)
        winner_output = assay_winner.run_outcome()
        outcome_array[i,1] = winner_output

        assay_loser = Fight(assay_fish,focal_loser,params)
        loser_output = assay_loser.run_outcome()
        outcome_array[i,0] = loser_output

        win_info_array[i] = focal_winner.effort,focal_winner.estimate 
        loss_info_array[i] = focal_loser.effort,focal_loser.estimate 

        #if assay_fish.wager > focal_winner.wager:
        #    w_probs.append(assay_winner.p_win)
        #else:
        #    w_probs.append(1-assay_winner.p_win)

        #if assay_fish.wager > focal_loser.wager:
        #    l_probs.append(assay_loser.p_win)
        #else:
        #    l_probs.append(1-assay_loser.p_win)


    return outcome_array,win_info_array,loss_info_array


iterations = 1000
params = Params()
params.iterations = iterations

print(params.awareness)
print(params.outcome_params)
params.size = 50

assay_params = params.copy()
#assay_params.baseline_effort = 0.535
#assay_params.prior = True

outcome_array = np.empty([iterations,2])
outcome_array.fill(np.nan)

win_info_array = np.array(outcome_array)
loss_info_array = np.array(outcome_array)
w_probs,l_probs = [],[]

f_sizes = []

s_res = 10+1
s_params = np.linspace(0,1.00,s_res)

win_estimates = np.empty([s_res,2])
win_efforts = np.empty_like(win_estimates)
win_outputs = np.empty_like(win_estimates)

loss_estimates = np.empty_like(win_estimates)
loss_efforts = np.empty_like(win_estimates)
loss_outputs = np.empty_like(win_estimates)

for s_ in range(len(s_params)):
    s = s_params[s_]
    print(s)
    params.outcome_params[0] = s
    params.set_params()

    outcome_array,win_info_array,loss_info_array = run_sim(params)
    """
    for i in tqdm(range(iterations)):
        focal_winner = Fish(i+2,params)
        focal_loser = focal_winner.copy() 

        f_sizes.append(focal_winner.size)
## Stage a bunch of wins and losses against size-matched fish

        staged_opp = Fish(0,params)
        
        staged_win = Fight(staged_opp,focal_winner,params,outcome=1)
        staged_win.run_outcome()
        focal_winner.update(True,staged_win)

        staged_loss = Fight(staged_opp,focal_loser,params,outcome=0)
        staged_loss.run_outcome()
        focal_loser.update(False,staged_loss)

## Assay against size matched fish
        assay_fish = Fish(1,assay_params)

        assay_winner = Fight(assay_fish,focal_winner,params)
        winner_output = assay_winner.run_outcome()
        outcome_array[i,1] = winner_output

        assay_loser = Fight(assay_fish,focal_loser,params)
        loser_output = assay_loser.run_outcome()
        outcome_array[i,0] = loser_output

        win_info_array[i] = focal_winner.effort,focal_winner.estimate 
        loss_info_array[i] = focal_loser.effort,focal_loser.estimate 

        #if assay_fish.wager > focal_winner.wager:
        #    w_probs.append(assay_winner.p_win)
        #else:
        #    w_probs.append(1-assay_winner.p_win)

        #if assay_fish.wager > focal_loser.wager:
        #    l_probs.append(assay_loser.p_win)
        #else:
        #    l_probs.append(1-assay_loser.p_win)
    """

## get win stats, using little helper function
    win_outputs[s_] = mean_sem(outcome_array[:,1])
    win_estimates[s_] = mean_sem(win_info_array[:,1])
    win_efforts[s_] = mean_sem(win_info_array[:,0])

    loss_outputs[s_] = mean_sem(outcome_array[:,0])
    loss_estimates[s_] = mean_sem(loss_info_array[:,1])
    loss_efforts[s_] = mean_sem(loss_info_array[:,0])

fig,axes = plt.subplots(3,1)
plot_fill(s_params,win_estimates,ax=axes[0],color='gold')
plot_fill(s_params,loss_estimates,ax=axes[0],color='darkblue')

plot_fill(s_params,win_efforts,ax=axes[1],color='gold')
plot_fill(s_params,loss_efforts,ax=axes[1],color='darkblue')

plot_fill(s_params,win_outputs,ax=axes[2],color='gold')
plot_fill(s_params,loss_outputs,ax=axes[2],color='darkblue')

axes[2].set_xlabel('s value')
axes[0].set_ylabel('Estimate')
axes[1].set_ylabel('Assay effort')
axes[2].set_ylabel('Assay win rate')

for ax in axes:
    ax.axvline(0.7,color='red',linestyle=':')
plt.show()
