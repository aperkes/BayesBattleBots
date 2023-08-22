#! /usr/bin/env python

## Script to show that under bayes', the duration of the winner effect depends on the about of new info 

from fish import Fish,FishNPC
from fight import Fight
from tank import Tank
from params import Params

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import random, copy

params = Params()

params.update_method = 'linear'
params.energy_cost = False
params.poly_param_b = 0
params.poly_param_m = 0.1
f = Fish(0,params)

opp_params = params.copy()
opp_params.size = f.size
#opp_params.effort_method = 'SmoothPoly'
#opp_params.baseline_effort = 0.25
npc_params = params.copy()
npc_params.baseline_effort = None

f2 = Fish(2,f.params)

matched_fish = Fish(1,opp_params)
#matched_fish = FishNPC(1,npc_params)
print('before:',f.estimate)
for n in range(100):
    if False:
        fight = Fight(matched_fish,f,params)
    elif n % 3:
        fight = Fight(matched_fish,f,params,outcome=0)
    else:
        fight = Fight(matched_fish,f,params,outcome=1)
    fight.run_outcome()
    f.update(fight.outcome,fight)
    #print(fight.outcome)
    #print('after:',f.estimate)

fig,ax = plt.subplots()
ax.plot(f.est_record)    
ax.axhline(f.size,color='black')
plt.show()

