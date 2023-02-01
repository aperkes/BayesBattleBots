
from fish import Fish,FishNPC
from fight import Fight
from tank import Tank
from params import Params

import numpy as np
from matplotlib import pyplot as plt


params = Params(outcome_params=[0.6,0.3,0.01])
params.energy_cost = False
params.awareness = 1
npc_params = params.copy()
all_errors = np.zeros([100,101])
for f in range(100):
    focal_fish = Fish(f,params)
    for i in range(100):
        npc = FishNPC(i+1,npc_params)
        fight = Fight(focal_fish,npc,params) 
        fight.run_outcome()
        focal_fish.update(1-fight.outcome,fight)
    all_errors[f] = np.array(focal_fish.est_record) - focal_fish.size
print('last one:',np.array(focal_fish.est_record) - focal_fish.size)
print('mean:',np.mean(all_errors,0))
print('std:',np.std(all_errors,0))

if True:
    fig,ax = plt.subplots()
    ax.axhline(0,linestyle=':',color='blue')
    for f in range(100):
        ax.plot(all_errors[f],alpha=.1,color='black')
    ax.set_ylim([-30,30])
    fig.savefig('./figures/fig5a_' + str(params.awareness) + '.png',dpi=300)    

print('done!')
