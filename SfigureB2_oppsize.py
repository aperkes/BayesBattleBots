
import numpy as np
from matplotlib import pyplot as plt

from bayesbots import Fish,FishNPC
from bayesbots import Fight
from bayesbots import Params

from tqdm import tqdm

## A couple of helper functions to keep things tidy

def mean_sem(a):
    mean_a = np.nanmean(a)
    sem_a = np.nanstd(a) / np.sqrt(len(a))
    return mean_a,sem_a

def plot_fill(xs,a,fig=None,ax=None,color='grey',alpha=0.5):
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(xs,a[:,0],color='black')
    ax.fill_between(xs,a[:,0] - a[:,1],a[:,0] + a[:,1],color=color,alpha=alpha)
    return fig,ax

def run_sim(params,rand_prior = False):
    iterations = params.iterations

    outcome_array = np.empty([iterations,2])
    outcome_array.fill(np.nan)

    win_info_array = np.array(outcome_array)
    loss_info_array = np.array(outcome_array)

    for i in tqdm(range(iterations)):
        if rand_prior:
            params.prior = np.random.randint(1,100)
        focal_winner = Fish(i+2,params)
        focal_loser = focal_winner.copy() 

        #f_sizes.append(focal_winner.size)
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


iterations = 3
params = Params()
params.iterations = iterations

params.size = 50

assay_params = params.copy()

print(params.awareness)
print(params.outcome_params)

#assay_params.baseline_effort = 0.535
#assay_params.prior = True

outcome_array = np.empty([iterations,2])
outcome_array.fill(np.nan)

win_info_array = np.array(outcome_array)
loss_info_array = np.array(outcome_array)
w_probs,l_probs = [],[]

f_sizes = [1,20,40,50,60,80,100]

size_res = len(f_sizes)

win_estimates = np.empty([2,size_res,2])
win_efforts = np.empty_like(win_estimates)
win_outputs = np.empty_like(win_estimates)

loss_estimates = np.empty_like(win_estimates)
loss_efforts = np.empty_like(win_estimates)
loss_outputs = np.empty_like(win_estimates)

for p_ in range(2): ## it's just centered priors and random priors
    params = Params()
    params.iterations = iterations
    assay_params = params.copy()

    if p_ == 0:
        rand_prior = False
    else:
        rand_prior = True
    for i_ in range(len(f_sizes)):
        i = f_sizes[i_]
        params.size = i
        assay_params.size = i

        #params = set_custom_params(params,p,i)
        outcome_array,win_info_array,loss_info_array = run_sim(params,rand_prior=rand_prior)
## get win stats, using little helper function
        win_outputs[p_,i_] = mean_sem(outcome_array[:,1])
        win_estimates[p_,i_] = mean_sem(win_info_array[:,1])
        win_efforts[p_,i_] = mean_sem(win_info_array[:,0])

        loss_outputs[p_,i_] = mean_sem(outcome_array[:,0])
        loss_estimates[p_,i_] = mean_sem(loss_info_array[:,1])
        loss_efforts[p_,i_] = mean_sem(loss_info_array[:,0])

fig,axes = plt.subplots(3,2)

#params_list = 

for p_ in range(2):
    rand_prior = p_
    xs_params = f_sizes
    plot_fill(xs_params,win_estimates[p_],ax=axes[0,p_],color='gold')
    plot_fill(xs_params,loss_estimates[p_],ax=axes[0,p_],color='darkblue')

    plot_fill(xs_params,win_efforts[p_],ax=axes[1,p_],color='gold')
    plot_fill(xs_params,loss_efforts[p_],ax=axes[1,p_],color='darkblue')

    plot_fill(xs_params,win_outputs[p_],ax=axes[2,p_],color='gold')
    plot_fill(xs_params,loss_outputs[p_],ax=axes[2,p_],color='darkblue')

axes[2,0].set_xlabel('size of focal agent \n(median-centered random prior)')
axes[2,1].set_xlabel('size of focal agent \n(size-centered prior)')

axes[0,0].set_ylabel('Estimate')
axes[1,0].set_ylabel('Assay effort')
axes[2,0].set_ylabel('Assay win rate')

for c_ in range(2):
    for r_ in range(3):
        ax = axes[r_,c_]
        ax.axvline(50,color='red',linestyle=':')

plt.show()
