## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np

#s_space = [0.0,.25,.5,.75,1.0]
#e_space = [0.0,.25,.5,.75,1.0]
#l_space = [0.0,0.1,0.5,0.9,1.0]
s_space = [0.0,0.2,0.4,0.6,0.8,1.0]
e_space = [0.0,0.2,0.4,0.6,0.8,1.0]
l_space = [0.0,.05,0.1,0.2,0.3,0.5]
results_array = np.full([6,6,6,3],np.nan)
params = SimParams()
params.effort_method = [0,1]
params.n_fights = 10*50
params.n_iterations = 15 
params.n_fish = 7
params.f_method = 'random'

for s_i in range(6):
    for e_j in range(6):
        for l_k in range(6):
            s = s_space[s_i]
            e = e_space[e_j]
            l = l_space[l_k]
            params.outcome_params = [s,e,l]
            fishes = [Fish(n,effort_method=params.effort_method) for n in range(7)]

            print('Working on',params.outcome_params)
            sim = Simulation(params)
            all_stats = np.array(sim.run_simulation())
            results_array[s_i,e_j,l_k] = np.mean(all_stats,axis=0)
            print(np.mean(all_stats,axis=0))
if False:
    all_stats = s.run_simulation()
    print(all_stats)
    all_stats = np.array(all_stats)
    print(np.mean(all_stats,axis=0))
elif False:
    tank = Tank(fishes,n_fights = 100,f_params=params.outcome_params)
    tank.run_all()
    lin,p = s._calc_linearity(tank)
    stab = s._calc_stability(tank)
    accu = s._calc_accuracy(tank)
    print(stab)

else:
    print('all done!')
    np.save('results_array.npy',results_array)
