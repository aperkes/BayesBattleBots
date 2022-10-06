## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from matplotlib import pyplot as plt

import numpy as np
import copy

#s_space = [0.0,.25,.5,.75,1.0]
#e_space = [0.0,.25,.5,.75,1.0]
#l_space = [0.0,0.1,0.5,0.9,1.0]
s_space = [0.0,0.2,0.4,0.6,0.8,1.0]
e_space = [0.0,0.2,0.4,0.6,0.8,1.0]
l_space = [0.0,.05,0.1,0.2,0.3,0.5]
results_array = np.full([6,6,6,3],np.nan)
params = SimParams()
params.effort_method = [1,1]
params.n_fights = 10 
params.n_iterations = 1000 
params.n_fish = 5
params.f_method = 'shuffled'
HOCK = False

if HOCK:
    params.u_method = 'hock'
    params.f_outcome = 'hock'
    params.outcome_params = [0.6,0.3,.05]
else:
    params.u_method = 'bayes'
    params.f_outcome = 'math'
    params.outcome_params = [1,0.1,.1]

s = Simulation(params)
## Check whether simulation is working and print stats
if False:
    print(params.u_method,params.f_outcome,params.outcome_params)
    all_stats = s.run_simulation(True)
    #print(all_stats)
    all_stats = np.array(all_stats)
    print('p-linearity | stability | accuracy | fight intensity')
    print(np.mean(all_stats,axis=0))
    print(np.std(all_stats,axis=0))

## Check whether the tank is working and plot this history
elif False:
    f = Fish(prior=True,fight_params=params.outcome_params)
    f2 = copy.deepcopy(f)
    matched_fight = Fight(f,f2,outcome=0,outcome_params=params.outcome_params)
    fig,ax = plt.subplots()
    print(matched_fight.mechanism,matched_fight.params)
    matched_fight.run_outcome()
    likelihood = f._use_mutual_likelihood(matched_fight)
    ax.plot(f.xs,likelihood,color='tab:orange')
    fig.savefig('./imgs/naive_likelihood.svg')
    fig2,ax2 = plt.subplots()
    ax2.plot(f.xs,f.prior,color='black')
    f.update(True,matched_fight)
    ax2.plot(f.xs,f.prior,color='gold')
    fig2.savefig('./imgs/prior.svg')
    fig.show()
    plt.show()

elif True:
    if True:
        pilot_fish = Fish(0,effort_method=params.effort_method,fight_params=params.outcome_params)
        fishes = [Fish(f,prior=True,likelihood = pilot_fish.naive_likelihood,fight_params=params.outcome_params,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]

    else:
        fishes = [Fish(f,prior=True,effort_method=params.effort_method,update_method=params.u_method) for f in range(params.n_fish)]


    tank = Tank(fishes,n_fights = params.n_fights,f_params=params.outcome_params,f_outcome=params.f_outcome,f_method=params.f_method,u_method=params.u_method,fitness_ratio=0.1,death=False)
    tank._initialize_likelihood()

    #ax.plot(fishes[0].xs,fishes[0].prior * 10,color='green')
    tank.run_all()
    #for f in fishes:
    #    ax.plot(f.xs,f.prior * 10)
    #ax.plot(f.xs,f.naive_likelihood,color='black')
    #fig.show()
    #plt.show()
    #lin,p = s._calc_linearity(tank)
    #stab = s._calc_stability(tank)
    #accu = s._calc_accuracy(tank)
    #print(stab,lin,accu)
    fig,ax = tank.plot_estimates(food=False)
    if False:
        for f in fishes:
            ax.plot(f.size_record)
    #tank.estimates = s._calc_dominance(tank)
    #print('dominance estimates:',tank.estimates)
    print('sizes:',tank.sizes)
    #print(np.arange(len(fishes))[np.argsort(tank.estimates)])
    print(np.arange(len(fishes))[np.argsort(tank.sizes)])
    ax.set_ylim([0,100])
    print([f.energy for f in fishes])
    print([f.alive for f in fishes])
    print(fishes[0].size_record)
    print('energy record:',fishes[0].energy_record)
    print('effort record:',fishes[0].win_record)
    print([sum(f.fitness_record) for f in fishes])
    if True:
        if False:
            if params.u_method == 'size_boost':
                fig.savefig('./imgs/fixed_estimates.jpg',dpi=300)
                fig.savefig('./imgs/fixed_estimates.svg')
            else:
                fig.savefig('./imgs/bayes_estimates.svg')
        fig.show()

        if False:
            fig2,ax2 = tank.plot_effort()

        effort_record = [f.level for f in tank.fight_list]
        upset_count = 0
        for f in tank.fight_list:
            try:
                if f.winner.size < f.loser.size:
                    upset_count += 1
            except:
                print('fight missing, come back to check this')

        print(upset_count / len(tank.fight_list))

        #print(effort_record) 
        if False:
            fig3,ax3 = plt.subplots()
            ax3.scatter(range(len(effort_record)),effort_record,alpha=.1) 
            fig3.show()

        fig4,ax4 = plt.subplots()
        ax4.plot(fishes[0].xs,fishes[0].prior)
        plt.show()
## Check whether the fights are working:
elif True:
    params.update_method = 'bayes'
    fishes = [Fish(f,effort_method=params.effort_method,update_method=params.update_method) for f in range(2)] ## Make two fish
    f1,f2 = fishes
    print(f1.summary(False))
    print(f2.summary(False))
    fight = Fight(f1,f2)
    fight.run_outcome()
    print('winner:',fight.winner.idx)
    fight.winner.update(True,fight)
    fight.loser.update(False,fight)
    print(fight.winner.effort,fight.winner.size,fight.winner.wager)
    print(f1.summary(False))
    print(f1.boost,f2.boost)
    print(f1.hock_estimate,f2.hock_estimate)
    print('that is all for now')
    


else:
    print('all done!')
    np.save('results_array.npy',results_array)
