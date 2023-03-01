
from fish import Fish,FishNPC
from fight import Fight
from tank import Tank
from params import Params

import numpy as np
from matplotlib import pyplot as plt


params = Params(outcome_params=[0.6,0.3,0.01])
params.energy_cost = False
params.save_me = True
params.plot_me = False
npc_params = params.copy()
params.awareness = 5
n_fights = 300
n_fish = 100
fig0,ax0 = plt.subplots()
for n_npcs in [1,5,n_fights]:
    print('running for:',n_npcs)
    all_errors = np.zeros([n_fish,n_fights + 1])
    for f in range(n_fish):
        focal_fish = Fish(f,params)
        npcs = [FishNPC(i+1,npc_params) for i in range(n_npcs)]
        for i in range(n_fights):
            pre_estimate = focal_fish.estimate
            i_ = i % n_npcs ## Reminder 1 % 2 = 1, 2 % 2 = 0
            #npc = FishNPC(i+1,npc_params)
            npc = npcs[i_]
            fight = Fight(focal_fish,npc,params) 
            fight.run_outcome()
            focal_fish.update(1-fight.outcome,fight)
            post_estimate = focal_fish.estimate
        all_errors[f] = np.array(focal_fish.est_record) - focal_fish.size
    #print('last one:',np.array(focal_fish.est_record) - focal_fish.size)
    #print('mean:',np.mean(all_errors,0))
    #print('std:',np.std(all_errors,0))

    mean_error = np.mean(np.abs(all_errors),0)
    sem_error = np.std(np.abs(all_errors),0) / np.sqrt(n_fish)
    xs = np.arange(len(mean_error))
    ax0.plot(xs,mean_error,label=n_npcs)
    ax0.fill_between(xs,mean_error - sem_error,mean_error + sem_error,alpha=.3,color='gray')
    if params.plot_me or params.save_me:
        fig,ax = plt.subplots()
        ax.axhline(0,linestyle=':',color='blue')
        for f in range(n_fish):
            ax.plot(all_errors[f],alpha=.1,color='black')
        ax.set_ylim([-30,30])
        if params.save_me:
            fig.savefig('./figures/fig5c_' + str(n_npcs) + '.png',dpi=300)    

fig0.legend()
if params.save_me:
    fig0.savefig('./figures/fig5c_all.png',dpi=300)
if params.plot_me:
    plt.show()
print('done!')
