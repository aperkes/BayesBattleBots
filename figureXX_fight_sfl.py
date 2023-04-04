#! /usr/bin/env python

## Script to iteration through the s,f,l, parameter space 
## showing who would win in a fight between two fish

import numpy as np
from tqdm import tqdm

from params import Params
from fish import Fish
from fight import Fight

## there's a transform here, so for the exponentional
## -1 => infinity 
## 0 => 1
## 1 == 0  

def build_params(s,f,l):
    fight_params = Params()
    fight_params.outcome_params = [s,f,l]
    fight_params.set_params()
    return fight_params

iterations=2
scale=21
p_space = np.linspace(-1,1,scale)
p_array = np.zeros([scale,scale,scale])
s,f,l = [0,0,0]

f1_params = Params()
f2_params = f1_params.copy()

f1_params.size = 60
f1_params.effort_method = [None,0.3]
f1_params.baseline_effort = 0.3

f2_params.size = 50
f2_params.effort_method = [None,0.6]
f2_params.baseline_effort = 0.6

f1 = Fish(0,f1_params)
f2 = Fish(1,f2_params)


for s_ in tqdm(range(scale)):
    s = p_space[s_]
    for f_ in range(scale):
        f = p_space[f_]
        for l_ in range(scale):
            l = p_space[-(1 + l_)]

            fight_params = build_params(s,f,l)

            fight = Fight(f1,f2,fight_params)
            outcome = fight.run_outcome()
            #import pdb;pdb.set_trace()
            if fight.f_min == 1:
                p_array[s_,f_,l_] = fight.p_win
            else:
                p_array[s_,f_,l_] = 1 - fight.p_win


print(np.round(p_array[:,0,:],3))
np.save('./results/fight_slf.npy',p_array)

print('done!')
