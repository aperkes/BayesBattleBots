
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

## Stupid little function to extract blocks of true values from an array
def get_chunks(x_bool):
    start = None 
    i_list = []
    all_is = []
    for i_ in range(len(x_bool)):
        if x_bool[i_] == False:
            if start is not None:
                all_is.append(i_list)
                i_list = []
                start = None
        else:
            if start is None:
                start = i_
            i_list.append(i_)
    if start is not None:
        all_is.append(i_list)
    return all_is

def plot_fill(xs,a,fig=None,ax=None,color='grey',alpha=0.5):
    if ax is None:
        fig,ax = plt.subplots()
    ax.plot(xs,a[:,0],color='black')
    top_line = a[:,0] + a[:,1]
    bottom_line = a[:,0] - a[:,1]
    if 'STAT' in color:
        #import pdb;pdb.set_trace()
        gold_points = bottom_line > 0
        blue_points = top_line < 0

        new_xs = np.linspace(min(xs),max(xs),len(xs) * 20)
        new_bottom = np.interp(new_xs,xs,bottom_line)
        new_top = np.interp(new_xs,xs,top_line)

        xs = new_xs
        if color == 'STAT':

            gray_bool = (new_bottom <= 0) & (new_top >= 0)
            gold_bits = get_chunks(new_bottom > 0)
            blue_bits = get_chunks(new_top < 0)
            gray_bits = get_chunks(gray_bool)
            for c in gray_bits:
                ax.fill_between(xs[c],new_bottom[c],new_top[c],color='gray',alpha=alpha)
            for c in gold_bits:
                ax.fill_between(xs[c],new_bottom[c],new_top[c],color='gold',alpha=alpha)
            for c in blue_bits:
                ax.fill_between(xs[c],new_bottom[c],new_top[c],color='darkblue',alpha=alpha)
        elif color == 'STAT_SIMPLE':
            gold_bits = get_chunks(new_top > 0)
            blue_bits = get_chunks(new_bottom < 0)
            for c in gold_bits:
                ax.fill_between(xs[c],np.clip(new_bottom[c],0,a_max=None),new_top[c],color='gold',alpha=alpha)
            for c in blue_bits:
                ax.fill_between(xs[c],new_bottom[c],np.clip(new_top[c],None,0),color='darkblue',alpha=alpha)
    else:
        ax.fill_between(xs,a[:,0] - a[:,1],a[:,0] + a[:,1],facecolor=color,alpha=alpha)
    y_max = max([np.max(np.abs(a[:,0] - a[:,1])),np.max(np.abs(a[:,0] + a[:,1]))])
    ax.set_ylim([y_max*-1.1,y_max*1.1])
   
    return fig,ax

def run_sim(params,opp_params):
    iterations = params.iterations

    outcome_list = [0,0]
    #outcome_array = np.empty([iterations,2])
    #outcome_array.fill(np.nan)

    #win_info_array = np.array(outcome_array)
    #loss_info_array = np.array(outcome_array)

    #for i in tqdm(range(iterations)):
    focal_winner = Fish(2,params)
    focal_loser = focal_winner.copy() 

## Stage a bunch of wins and losses against size-matched fish

    staged_opp = FishNPC(0,opp_params)
    
    staged_win = Fight(staged_opp,focal_winner,params,outcome=1)
    staged_win.run_outcome()
    focal_winner.update(True,staged_win)

    staged_loss = Fight(staged_opp,focal_loser,params,outcome=0)
    staged_loss.run_outcome()
    focal_loser.update(False,staged_loss)

## Assay against size matched fish
    assay_fish = Fish(1,params)

    assay_winner = Fight(assay_fish,focal_winner,params)
    winner_output = assay_winner.run_outcome()
    outcome_list[1] = winner_output

    assay_loser = Fight(assay_fish,focal_loser,params)
    loser_output = assay_loser.run_outcome()
    outcome_list[0] = loser_output

    win_info = focal_winner.effort,focal_winner.estimate 
    loss_info = focal_loser.effort,focal_loser.estimate 

    return outcome_list,win_info,loss_info


#iterations = 1000
params = Params()
#params.iterations = iterations

params.size = 50

#assay_params = params.copy()

#assay_params.baseline_effort = 0.535
#assay_params.prior = True


s_res = 10+1
s_params = np.linspace(0,1.00,s_res)

default_params = [params.outcome_params[0],params.outcome_params[2],params.awareness,params.acuity]

e_res = 11
s_res = 11
opp_efforts = np.linspace(0,1,e_res)
opp_sizes = np.linspace(1,100,s_res)

params = Params()
params.size = 50
params.acuity = 0
params.prior = True
params.set_params()

opp_params = Params()

est_winners = np.zeros([s_res,e_res])
est_losers = np.zeros([s_res,e_res])

for e_ in range(len(opp_efforts)):
    for s_ in range(len(opp_sizes)):
        s = opp_sizes[s_]
        eff = opp_efforts[e_]
        opp_params.size = s
        opp_params.baseline_effort = eff

        outcomes,win_info,loss_info = run_sim(params,opp_params)

        est_winners[e_,s_] = win_info[1]
        est_losers[e_,s_] = loss_info[1]

fig,axes = plt.subplots(1,2,sharey=True,sharex=True)

vmax = np.max([est_winners,est_losers])
vmin = np.max([est_winners,est_losers])
max_dist = max(np.abs([vmax-50,vmin-50]))
vmax = 50 + max_dist
vmin = 50 - max_dist

axes[0].imshow(est_winners,vmax=vmax,vmin=vmin,cmap='RdBu_r')
im = axes[1].imshow(est_losers,vmax=vmax,vmin=vmin,cmap='RdBu_r')

axes[0].set_ylabel('Opponent Effort')
axes[0].set_xlabel('Opponent Size')
axes[1].set_xlabel('Opponent Size')

ax1,ax2 = axes
ax1.set_xticks(range(s_res))

ax1.set_xticklabels(np.round(opp_sizes,2).astype(int),rotation=45)
ax2.set_xticklabels(np.round(opp_sizes,2).astype(int),rotation=45)

ax1.set_yticks(range(e_res))
ax1.set_yticklabels(np.round(opp_efforts,1))

fig.colorbar(im,ax=axes,shrink=0.50)

plt.show()
