## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from matplotlib import pyplot as plt

import numpy as np

#s_space = [0.0,.25,.5,.75,1.0]
#e_space = [0.0,.25,.5,.75,1.0]
#l_space = [0.0,0.1,0.5,0.9,1.0]
s_space = [0.0,0.2,0.4,0.6,0.8,1.0]
e_space = [0.0,0.2,0.4,0.6,0.8,1.0]
l_space = [0.0,.05,0.1,0.2,0.3,0.5]
results_array = np.full([6,6,6,3],np.nan)
params = SimParams()
params.effort_method = [1,1]
params.n_fights = 10*50
params.n_iterations = 1000 
params.n_fish = 7
params.f_method = 'random'

HOCK = True

params.outcome_params = [0.6,0.3,.05]

if False:
    params.u_method = 'hock'
    params.f_outcome = 'hock'
    #params.outcome_params = [0.6,0.3,.05]
else:
    params.f_outcome = 'math'

    if False:
        params.u_method = 'bayes'
        #params.outcome_params = [0.6,0.3,.05]
    elif True:
        params.u_method = 'decay'
    else:
        params.u_method = 'fixed'

s = Simulation(params)
## Check whether simulation is working and print stats

## Check whether the tank is working and plot this history
accuracies = []
all_stats = np.array(s.run_simulation())
stab,lin,acc = np.mean(all_stats,axis=0)
s_,l_,acc_std = np.std(all_stats,axis=0)

print('Decay WE')
print(acc,acc_std / np.sqrt(params.n_iterations))
