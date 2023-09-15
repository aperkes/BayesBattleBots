
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
    focal_WW = Fish(1,params)
    focal_LL = Fish(2,params)

## This only works if acuity is 0 and prior = True
    staged_opp = Fish(0,params)
    for i in range(iterations):
        focal_WL = copy.deepcopy(focal_WW)
        focal_LW = copy.deepcopy(focal_LL)
        #f_sizes.append(focal_winner.size)
## Stage a bunch of wins and losses against size-matched fish
        W_l = Fight(staged_opp,focal_WL,params,outcome=0) 
        W_l.run_outcome()
        focal_WL.update(False,W_l)
        
        L_w = Fight(staged_opp,focal_LW,params,outcome=1) 
        L_w.run_outcome()
        focal_LW.update(True,L_w)

        #print(params.outcome_params,params.awareness,i,focal_WL.estimate,focal_LW.estimate)
        if focal_WL.estimate >= focal_LW.estimate:
            return i

        W_w = Fight(staged_opp,focal_WW,params,outcome=1)
        W_w.run_outcome()
        focal_WW.update(True,W_w)

        L_l = Fight(staged_opp,focal_LL,params,outcome=0) 
        L_l.run_outcome()
        focal_LL.update(False,L_l)

    return iterations

#iterations = 2
iterations = 100
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

n_repeats = np.zeros([s_res,a_res,l_res])

for s_ in tqdm(range(s_res)):
    for a_ in range(a_res):
        for l_ in range(l_res):
            params.outcome_params = [s_set[s_],0.5,l_set[l_]]
            params.awareness = a_set[a_]
            params.set_params()

            n_repeats[s_,a_,l_] = run_sim(params)

v_max = np.nanmax(n_repeats) 
## Plot stuff
repeats_l8 = n_repeats[:,:,1]

#fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
fig,ax = plt.subplots()

im = ax.imshow(repeats_l8,vmin=0,vmax=v_max)
#im1 = axes[1].imshow(n_repeats[:,:,5],vmin=0,vmax=v_max)
#im2 = axes[2].imshow(n_repeats[:,:,9],vmin=0,vmax=v_max)

fig.colorbar(im,ax=ax)

axes = [ax]
axes[0].set_xticks(range(s_res))
axes[0].set_yticks(range(a_res))

axes[0].set_xticklabels(np.round(a_set,2))
axes[0].set_yticklabels(np.round(s_set,2))

axes[0].set_xlabel('Awareness value')
axes[0].set_ylabel('s value')

plt.show()
print('all done, do you want to check anything?')

#import pdb;pdb.set_trace()
print('Done.')
