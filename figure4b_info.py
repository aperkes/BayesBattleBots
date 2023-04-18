#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish,FishNPC
from fight import Fight
from tank import Tank
from simulation import Simulation
from params import Params

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import ttest_ind

from tqdm import tqdm

import copy

#s,e,l = -.8,-.6,-0.99
#s,e,l=-0.9,-0.5,-0.7

params = Params()
params.mean_size = 50
params.sd_size = 20
#params.poly_param_c = 0
#params.effort_method = [1,1]
params.f_method = 'shuffled'
params.n_rounds = 35 
params.n_fish = 5
params.acuity = 0.5 
params.awareness = 0.5
#params.outcome_params = [s,e,l]
params.set_params()
params.baseline_effort = 0.3

replicates = 100

#opp_sizes = [45,50,65] ## size of staged fight
#tank_sizes = [35,45,65,75] ## size of other fish
opp_size = 65 
tank_sizes = [3,5,10,20,40]
cutoff = 100

params_f0 = params.copy()
params_f0.prior = True
params_f0.size = 50
#f0 = Fish(0,params_f0)

n_points = cutoff + 2
#n_points =  params.n_rounds * (params.n_fish - 1) + 2
est_array = np.empty([len(tank_sizes),replicates,n_points])
est_array.fill(np.nan)


for n_ in range(len(tank_sizes)):
    #other_fishes = [Fish(f,params) for f in range(tank_sizes[n_])]
    n = tank_sizes[n_]
    params.n_rounds = 100 // (n - 1) + 1
    for i in tqdm(range(replicates)):
        #opp_size = 
        #fishes = copy.deepcopy(other_fishes)
        fishes = [Fish(f,params) for f in range(tank_sizes[n_])]
        f0 = Fish(0,params_f0)
        fishes[0] = f0
        focal_fish = f0
        tank = Tank(fishes,params)

        ## Before running the tank, simulate a win: 
        #focal_fish = fishes[-1]
        f_opp = FishNPC(0,focal_fish.params)
        f_opp.size = opp_size
        focal_win = Fight(f_opp,focal_fish,params,outcome=1)
        focal_win.run_outcome()
        focal_fish.update(True,focal_win)


        opp_indices = np.arange(1,tank_sizes[n_]-1)
        opps = np.random.choice(opp_indices,100)
        """
        for o_ in range(len(opps)):
            opponent = fishes[opps[o_]]
            fight = Fight(opponent,focal_fish,params)
            #print(opponent.effort,focal_fish.effort)
            outcome = fight.run_outcome()
            focal_fish.update(outcome,fight)
            opponent.update(1-outcome,fight)
        """
## Run tank and save est_record
        tank.run_all(progress=False)

        f_est = np.array(focal_fish.est_record)[:102]
        est_array[n_,i,:len(focal_fish.est_record)] = f_est

fig,ax = plt.subplots()
for s_ in range(len(tank_sizes)):
    est_mean = np.mean(est_array[s_],axis=0)
    est_sem = np.std(est_array[s_],axis=0) /  np.sqrt(replicates)
    xs = np.arange(len(est_mean))
    cor = cm.viridis((s_+1) / len(tank_sizes))
    ax.plot(xs,est_mean,color='black')
    ax.fill_between(xs,est_mean-est_sem,est_mean+est_sem,alpha=0.5,color=cor,label=tank_sizes[s_])
    for t in np.arange(tank_sizes[s_]-1,100,tank_sizes[s_]-1):
        ax.scatter(t,est_mean[t-1],marker='o',color='black')

ax.axhline(50,color='black',linestyle=':')
ax.legend()

ax.set_ylabel('Self-Estimate')
ax.set_xlabel('n contests')
fig.show()
fig.savefig('./figures/fig4b_info.png',dpi=300)
plt.show()
