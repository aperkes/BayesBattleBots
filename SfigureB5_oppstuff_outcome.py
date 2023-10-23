
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from bayesbots import Fish,FishNPC
from bayesbots import Fight
from bayesbots import Params

from tqdm import tqdm
import copy
from joblib import Parallel, delayed

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

def run_sim(params,opp_params,npc_size_list = [50],npc_effort_list = [0.5]):
    iterations = params.iterations

## Fish WL, LW
    focal_winner = Fish(1,params)
    focal_loser = Fish(2,params)

## Cant pre run the first two fights against NPC if I want acuity
    staged_opp = FishNPC(0,opp_params)
    max_rounds = 100

    random_opp = FishNPC(1,opp_params)
    results = np.empty([iterations,2])
    for i in range(iterations):
        focal_W = copy.deepcopy(focal_winner)
        focal_L = copy.deepcopy(focal_loser)

        staged_win = Fight(staged_opp,focal_W,params,outcome=1)
        staged_win.run_outcome()
        focal_W.update(True,staged_win)

        staged_loss = Fight(staged_opp,focal_L,params,outcome=0)
        staged_loss.run_outcome()
        focal_L.update(False,staged_loss)

        #f_sizes.append(focal_winner.size)
## Stage a bunch of wins and losses against size-matched fish
        W_thresh = 60
        L_thresh = 40
        w_rounds,l_rounds = np.nan,np.nan
        npc_sizes = np.random.choice(npc_size_list,max_rounds)
        npc_efforts = np.random.choice(npc_effort_list,max_rounds)
        for n in range(max_rounds):
            if ~np.isnan(w_rounds) and ~np.isnan(l_rounds):
                break
            npc = random_opp
            npc.baseline_effort = npc_efforts[n]
            npc.size = npc_sizes[n]
            if np.isnan(w_rounds):
                if focal_W.estimate <= W_thresh:
                    w_rounds = n
                else:
## Run a random fight
                    random_fight = Fight(npc,focal_W,params) 
                    outcome = random_fight.run_outcome()
                    focal_W.update(outcome,random_fight)
            if np.isnan(l_rounds):
                if focal_L.estimate >= L_thresh:
                    l_rounds = n
                else:
                    random_fight = Fight(npc,focal_L,params)
                    outcome = random_fight.run_outcome()
                    focal_L.update(outcome,random_fight)
        #import pdb;pdb.set_trace()
        results[i] = w_rounds,l_rounds
    return results

#iterations = 2
iterations = 100
params = Params()
params.iterations = iterations

params.size = 50
params.acuity = 0
params.prior = True

opp_params = params.copy()

s_res = 10+1
e_res = s_res

s_set = np.linspace(0,100,s_res)
e_set = np.linspace(0,1,e_res)
e_set[0] = 0.01
s_set[0] = 1

default_params = [params.outcome_params[0],params.outcome_params[2],params.awareness,params.acuity]

all_results = np.empty([s_res,e_res,iterations,2])

def build_trunc(mu,std,lower,upper):
    X = stats.truncnorm((lower-mu)/std,(upper-mu)/std,loc=mu,scale=std)
    return X

mu_s,std_s = 50,10
lower_s,upper_s = 1,100

mu_e,std_e = 0.5,0.1
lower_e,upper_e = 0.001,1

## Using the truncated distribution prevents weird bunching up at edge
X_size = build_trunc(mu_s,std_s,lower_s,upper_s)
X_eff = build_trunc(mu_e,std_e,lower_e,upper_e)
npc_size_list = X_size.rvs(100)
npc_effort_list = X_eff.rvs(100)

def run_many_sims(s):
    some_results = np.empty([e_res,iterations,2])
    for e_ in tqdm(range(e_res)):
            opp_params.baseline_effort = e_set[e_]
            opp_params.size = s
            some_results[e_] = run_sim(params,opp_params,npc_size_list,npc_effort_list)
    return some_results

all_results = Parallel(n_jobs=11)(delayed(run_many_sims)(s) for s in s_set)
all_results = np.array(all_results)

"""
for s_ in tqdm(range(s_res)):
        for e_ in range(e_res):
            opp_params.baseline_effort = e_set[e_]
            opp_params.size = s_set[s_]
            #import pdb;pdb.set_trace()
            all_results[s_,e_] = run_sim(params,opp_params)
"""

mean_results = np.nanmean(all_results,axis=2)
mean_winners = mean_results[:,:,0]
mean_losers = mean_results[:,:,1]

v_max = np.nanmax(mean_results)

## Plot stuff

#fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,sharex=True)

ax1.imshow(mean_winners,vmin=0,vmax=v_max)
im = ax2.imshow(mean_losers,vmin=0,vmax=v_max)

fig.colorbar(im,ax=[ax1,ax2],shrink=0.5)

axes = [ax1,ax2]
axes[0].set_xticks(range(e_res))
axes[0].set_yticks(range(s_res))

axes[0].set_xticklabels(np.round(e_set,2),rotation=45)
axes[1].set_xticklabels(np.round(e_set,2),rotation=45)

axes[0].set_yticklabels(np.round(s_set,2).astype(int))

axes[0].set_xlabel('Opp effort')
axes[1].set_xlabel('Opp effort')
axes[0].set_ylabel('Opp size')

plt.show()
print('all done, do you want to check anything?')

#import pdb;pdb.set_trace()
print('Done.')
