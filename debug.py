## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np

params = SimParams()
params.outcome_params = [0.0,0.0,.5]
params.effort_method = [1,0]
params.n_fights = 100
params.n_iterations = 2
params.n_fish = 7

fishes = [Fish(n,effort_method=params.effort_method) for n in range(7)]
tank = Tank(fishes,n_fights = 100,f_params=params.outcome_params)
tank.run_all()

print(tank.win_record)
print(np.sum(tank.win_record))
print(tank.n_fights)
s = Simulation(params)
#all_stats = s.run_simulation()

if False:
    print(all_stats)
    all_stats = np.array(all_stats)
    print(np.mean(all_stats,axis=0))
else:
    lin,p = s._calc_linearity(tank)
    stab = s._calc_stability(tank)
    accu = s._calc_accuracy(tank)
    print(lin,p)
