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
params.u_method = 'hock'
params.f_outcome = 'hock'

s = Simulation()
if False:
    all_stats = s.run_simulation()
    print(all_stats)
    all_stats = np.array(all_stats)
    print(np.mean(all_stats,axis=0))
elif True:
    fishes = [Fish(f,effort_method=params.effort_method) for f in range(params.n_fish)]
    tank = Tank(fishes,n_fights = 100,f_params=params.outcome_params,f_outcome=params.f_outcome,f_method=params.f_method,u_method=params.u_method)
    tank.run_all()
    lin,p = s._calc_linearity(tank)
    stab = s._calc_stability(tank)
    accu = s._calc_accuracy(tank)
    print(stab,lin)
    fig,ax = tank.plot_estimates()
    fig.savefig('test.jpg',dpi=300)
else:
    print('all done!')
    np.save('results_array.npy',results_array)
