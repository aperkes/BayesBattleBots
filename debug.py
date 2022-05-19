## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation,SimParams

import numpy as np

params = SimParams()
params.outcome_params = [1.0,0.0,.0]
params.effort_method = [1,0]
params.n_fights = 100
params.n_iterations = 50
params.n_fish = 7

fishes = [Fish(n,effort_method=params.effort_method) for n in range(7)]

s = Simulation(params)

if True:
    all_stats = s.run_simulation()
    print(all_stats)
    all_stats = np.array(all_stats)
    print(np.mean(all_stats,axis=0))
else:
    tank = Tank(fishes,n_fights = 100,f_params=params.outcome_params)
    tank.run_all()
    lin,p = s._calc_linearity(tank)
    stab = s._calc_stability(tank)
    accu = s._calc_accuracy(tank)
    print(stab)
