## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from fight import Fight
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
HOCK = False

if HOCK:
    params.u_method = 'hock'
    params.f_outcome = 'hock'
    params.outcome_params = [1.0,0.3,.1]
else:
    params.u_method = 'bayes'
    params.f_outcome = 'math'
    params.outcome_params = [1.0,0.3,.1]
s = Simulation(params)
## Check whether simulation is working and print stats
if False:
    all_stats = s.run_simulation()
    print(all_stats)
    all_stats = np.array(all_stats)
    print(np.mean(all_stats,axis=0))

## Check whether the tank is working and plot this history
elif True:
    fishes = [Fish(f,effort_method=params.effort_method) for f in range(params.n_fish)]
    tank = Tank(fishes,n_fights = 1000,f_params=params.outcome_params,f_outcome=params.f_outcome,f_method=params.f_method,u_method=params.u_method)
    tank.run_all()
    lin,p = s._calc_linearity(tank)
    stab = s._calc_stability(tank)
    accu = s._calc_accuracy(tank)
    print(stab,lin)
    fig,ax = tank.plot_estimates()
    fig.savefig('test.jpg',dpi=300)

## Check whether the fights are working:
elif True:
    fishes = [Fish(f,effort_method=params.effort_method) for f in range(2)] ## Make two fish
    f1,f2 = fishes
    print(f1.summary(False))
    print(f2.summary(False))
    fight = Fight(f1,f2)
    fight.run_outcome()
    print('winner:',fight.winner.idx)
    fight.winner.update_prior_(True,fight)
    fight.loser.update_prior_(False,fight)
    print(fight.winner.effort,fight.winner.size,fight.winner.wager)
    print(f1.summary(False))
    print('that is all for now')
    


else:
    print('all done!')
    np.save('results_array.npy',results_array)
