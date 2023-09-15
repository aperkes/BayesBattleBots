
import numpy as np
from matplotlib import pyplot as plt

from fish import Fish,FishNPC
from fight import Fight
from params import Params

from tqdm import tqdm
import copy

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

def run_sim(params):
    iterations = params.iterations

    outcome_array = np.empty([iterations,2])
    outcome_array.fill(np.nan)

    WL_info_array = np.array(outcome_array)
    LW_info_array = np.array(outcome_array)

## Fish WL, LW
    focal_winner = Fish(1,params)
    focal_loser = Fish(2,params)

## Cant pre run the first two fights against NPC if I want acuity
    staged_opp = FishNPC(0,params)
    max_rounds = 100

    random_opp = FishNPC(1,params)
    for i in range(iterations):
        focal_W = copy.deepcopy(focal_winner)
        focal_L = copy.deepcopy(focal_loser)

        staged_win = Fight(staged_opp,focal_W,params,outcome=1)
        staged_win.run_outcome()
        focal_W.update(True,staged_win)

        staged_loss = Fight(staged_opp,focal_L,params,outcome=0)
        staged_loss.run_outcome()
        focal_L.update(True,staged_win)

        #f_sizes.append(focal_winner.size)
## Stage a bunch of wins and losses against size-matched fish
        W_thresh = 60
        L_thresh = 40
        w_rounds,l_rounds = np.nan,np.nan
        for n in range(max_rounds):
            if np.isnan(w_rounds) and np.isnan(l_rounds):
                break
            npc = random_opp
            npc.baseline_effort = 0.5
            npc.size = 50
            if np.isnan(w_rounds):
                if focal_W.estimate <= W_thresh
                    w_rounds = n
                else:
## Run a random fight
                    random_fight = Fight(npc,focal_W,params) 
                    outcome = random_fight.run_outcome()
                    focal_W.update(outcome,random_fight)
            if np.isnan(l_rounds):
                if focal_L.estimate >= W_thresh:
                    w_rounds = n
                else:
                    random_fight = Fight(npc,focal_L,params)
                    outcome = random_fight.run_outcome()
                    focal_L.update(outcome,random_fight)
        results[i] = w_rounds,l_rounds
    return results

#iterations = 2
iterations = 10
params = Params()
params.iterations = iterations

params.size = 50
params.acuity = 0
params.prior = True

assay_params = params.copy()

s_res = 10+1
l_res = s_res
a_res = s_res

s_set = np.linspace(0,1,s_res)
l_set = np.linspace(-1,1,l_res)
a_set = np.linspace(0,1,a_res)
s_set[0] = 0.01
a_set[0] = 0.01
#c_set = np.linspace(0,1,a_res)

default_params = [params.outcome_params[0],params.outcome_params[2],params.awareness,params.acuity]

all_results = np.empty([s_res,e_res,2])

for s_ in tqdm(range(s_res)):
        for e_ in range(e_res):
            params.outcome_params = [s_set[s_],0.5,l_set[l_]]
            params.awareness = a_set[a_]
            params.set_params()

            all_results[s_,e_] = run_sim(params)


mean_results = np.nanmean(all_results,axis=2)
mean_winners = all_results[:,:,0]
mean_losers = all_results[:,:,1]

v_max = np.nanmax(mean_results)

## Plot stuff

#fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,sharex=True)

ax.imshow(mean_winners,vmin=0,vmax=v_max)
im = ax.imshow(mean_losers,vmin=0,vmax=v_max)

fig.colorbar(im,ax=[ax1,ax2])

axes = [ax1,ax2]
axes[0].set_xticks(range(e_res))
axes[0].set_yticks(range(s_res))

axes[0].set_xticklabels(np.round(e_set,2))
axes[0].set_yticklabels(np.round(s_set,2))

axes[0].set_xlabel('Opp effort')
axes[0].set_ylabel('Opp size')

plt.show()
print('all done, do you want to check anything?')

#import pdb;pdb.set_trace()
print('Done.')
