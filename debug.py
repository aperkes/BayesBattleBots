## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np

s_space = [0.0,.25,.5,.75,1.0]
e_space = [0.0,.25,.5,.75,1.0]
l_space = [0.0,0.1,0.5,0.9,1.0]

results_array = np.empty([5,5,5,3])

params = SimParams()
params.effort_method = [1,0]
params.n_fights = 5
params.n_iterations = 5
params.n_fish = 7
params.f_method = 'balanced'

for s_i in range(5):
    for e_j in range(5):
        for l_k in range(5):
            s = s_space[s_i]
            e = e_space[e_j]
            l = l_space[l_k]
            if s == e and s != 0.0:
                results_array[s_i,e_j,l_k] = results_array[0,0,l_k]
            else:
                params.outcome_params = [s,e,l]
                fishes = [Fish(n,effort_method=params.effort_method) for n in range(7)]

                print('Working on',params.outcome_params)
                sim = Simulation(params)
                all_stats = np.array(sim.run_simulation())
                results_array[s_i,e_j,l_k] = np.mean(all_stats,axis=0)
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
    np.save('restuls_array.npy',results_array)
