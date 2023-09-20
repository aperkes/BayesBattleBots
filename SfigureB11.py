
from fish import Fish
from tank import Tank
from params import Params

import numpy as np
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

from joblib import Parallel, delayed
## Build repeated sim: 

def mean_sem(a):
    mean_a = np.nanmean(a)
    sem_a = np.nanstd(a) / np.sqrt(len(a))
    return mean_a,sem_a

def run_sim(params):
## Initialize a tank

    f_errors = np.empty([params.n_iterations,params.n_fish])
    f_initial = np.empty([params.n_iterations,params.n_fish])
    end_costs = np.empty_like(f_initial)
    init_costs = np.empty_like(f_initial)

    for i in range(params.n_iterations):
        fishes = [Fish(f,params) for f in range(params.n_fish)]

## Run the tank
        tank = Tank(fishes,params)

        tank.run_all(progress=False)

## Check accuracy (How) 
        for f_ in range(params.n_fish):
            f = tank.fishes[f_]

            last_rounds = [f.win_record[r_][3] for r_ in range(-params.n_fish + 1,0)]
            first_rounds = [f.win_record[r_][3] for r_ in range(0,params.n_fish)]
            initial_cost = np.mean(first_rounds)
            last_cost = np.mean(last_rounds)

            #print(f.size,f.estimate,f.sdest_record[-1])
            f_estimate = f.estimate
            f_estimate_0 = f.est_record[0]
            f_std_0 = f.sdest_record[0]

            f_std = f.sdest_record[-1]
            f_error = np.sqrt((f_estimate + f_std - f.size)**2 + (f_estimate - f_std - f.size)**2)
            f_error_0 = np.sqrt((f_estimate_0 + f_std_0 - f.size)**2 + (f_estimate_0 - f_std_0 - f.size)**2)
            #f_error = (f.size - f.estimate)**2 /(f.sdest_record[-1])

            #f_errors[i,f_] = f.sdest_record[-1]
            f_errors[i,f_] = f_error
            f_initial[i,f_] = f_error_0
            end_costs[i,f_] = last_cost
            init_costs[i,f_] = initial_cost
            #f_initial[i,f_] = (f.size - f.est_record[0])**2 / f.est_record[0]
    return f_errors,f_initial,end_costs,init_costs

## should I do the whole range? 

s_res = 11
l_res = s_res
a_res = s_res

s_list = np.linspace(0,1,s_res)
l_list = np.linspace(-1,1,l_res)
a_list = np.linspace(0,1,a_res)
c_list = np.linspace(0,1,a_res)


params = Params()
params.n_rounds = 10
params.n_iterations = 10

def run_many_sims(s,params):
    params = copy.deepcopy(params)
    some_errors = np.empty([l_res,a_res,a_res,8])
    for l_ in tqdm(range(l_res)):
        l = l_list[l_]
        params.outcome_params = [s,0.5,l]
        for a_ in range(a_res):
            a = a_list[a_]
            params.awareness = a
            for c_ in range(a_res):
                c = c_list[c_]
                params.acuity = c                
                params.set_params()
                f_errors,f_initial,end_costs,init_costs = run_sim(params)
                some_errors[l_,a_,c_,:2] = mean_sem(f_errors)
                some_errors[l_,a_,c_,2:4] = mean_sem(f_initial)
                some_errors[l_,a_,c_,4:6] = mean_sem(end_costs)
                some_errors[l_,a_,c_,6:8] = mean_sem(init_costs)

    return some_errors

if False:
    all_results = np.empty([s_res,l_res,a_res,a_res,8])
    for s_ in tqdm(range(s_res)):
        s = s_list[s_]
        all_results[s_] = run_many_sims(s,params)

else:
    all_results = Parallel(n_jobs=11)(delayed(run_many_sims)(s,params) for s in s_list)
    all_results = np.array(all_results)

all_errors = all_results[:,:,:,:,0:4]
all_costs = all_results[:,:,:,:,4:8]

mean_errors = all_errors[:,:,:,:,0]
sem_errors = all_errors[:,:,:,:,1]

mean_init = all_errors[:,:,:,:,2]
sem_init = all_errors[:,:,:,:,3]

mean_costs = all_costs[:,:,:,:,0]
sem_costs = all_costs[:,:,:,:,1]
init_costs = all_costs[:,:,:,:,2]
#import pdb; pdb.set_trace()

fig,axes = plt.subplots(1,4,sharey=True)

s_mean = mean_errors[:,1,5,1]
s_sem = sem_errors[:,1,5,1]

axes[0].plot(s_list,mean_errors[:,1,5,1])
axes[0].plot(s_list,mean_init[:,1,5,1],color='black',linestyle=':')

axes[0].fill_between(s_list,s_mean-s_sem,s_mean+s_sem,alpha=0.5,color='gray')
axes[0].set_xlabel('s value')

l_mean = mean_errors[7,:,5,1]
l_sem = sem_errors[7,:,5,1]
axes[1].plot(l_list,mean_errors[7,:,5,1])
axes[1].plot(l_list,mean_init[7,:,5,1],color='black',linestyle=':')
axes[1].fill_between(l_list,l_mean-l_sem,l_mean+l_sem,alpha=0.5,color='gray')

axes[1].set_xlabel('l value')

a_mean = mean_errors[7,1,:,1]
a_sem = sem_errors[7,1,:,1]
axes[2].plot(a_list,mean_errors[7,1,:,1])
axes[2].plot(a_list,mean_init[7,1,:,1],color='black',linestyle=':')
axes[2].fill_between(a_list,a_mean-a_sem,a_mean+a_sem,alpha=0.5,color='gray')

axes[2].set_xlabel('self assessment error')

c_mean = mean_errors[7,1,5,:]
c_sem = sem_errors[7,1,5,:]
axes[3].plot(c_list,mean_errors[7,1,5,:])
axes[3].plot(c_list,mean_init[7,1,5,:],color='black',linestyle=':')
axes[3].fill_between(c_list,c_mean-c_sem,c_mean+c_sem,alpha=0.5,color='gray')

axes[3].set_xlabel('opp assessment error')

axes[0].set_ylabel('Accuracy \n(std of estimate)')

fig2,axes2 = plt.subplots(1,4,sharey=True)
s_mean = mean_costs[:,1,5,1]
s_sem = sem_costs[:,1,5,1]
s_init = init_costs[:,1,5,1]

axes2[0].plot(s_list,s_mean)
axes2[0].plot(s_list,s_init,color='black',linestyle=':')

axes2[0].fill_between(s_list,s_mean-s_sem,s_mean+s_sem,alpha=0.5,color='gray')
axes2[0].set_xlabel('s value')

l_mean = mean_costs[7,:,5,1]
l_sem = sem_costs[7,:,5,1]
l_init = init_costs[7,:,5,1]

axes2[1].plot(l_list,l_mean)
axes2[1].plot(l_list,l_init,color='black',linestyle=':')
axes2[1].fill_between(l_list,l_mean-l_sem,l_mean+l_sem,alpha=0.5,color='gray')

axes2[1].set_xlabel('l value')

a_mean = mean_costs[7,1,:,1]
a_sem = sem_costs[7,1,:,1]
a_init = init_costs[7,1,:,1]

axes2[2].plot(a_list,a_mean)
axes2[2].plot(a_list,a_init,color='black',linestyle=':')
axes2[2].fill_between(a_list,a_mean-a_sem,a_mean+a_sem,alpha=0.5,color='gray')

axes2[2].set_xlabel('self assessment error')

c_mean = mean_costs[7,1,5,:]
c_sem = sem_costs[7,1,5,:]
c_init = init_costs[7,1,5,:]

axes2[3].plot(c_list,c_mean)
axes2[3].plot(c_list,c_init,color='black',linestyle=':')
axes2[3].fill_between(c_list,c_mean-c_sem,c_mean+c_sem,alpha=0.5,color='gray')

axes2[3].set_xlabel('opp assessment error')

axes2[0].set_ylabel('Mean cost of round')


## NOTE: Next thing to do is add linearity calculation in here too

plt.show()
