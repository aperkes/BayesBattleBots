
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

    end_lin = np.empty_like(f_initial)
    init_lin = np.empty_like(f_initial)

    end_stab = np.empty_like(f_initial)
    init_stab = np.empty_like(f_initial)
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
            init_idx = slice(0,3)
            final_idx = slice(-3,None)
            initial_linearity,_ = tank._calc_linearity(init_idx)
            last_linearity,_ = tank._calc_linearity(final_idx)

            initial_stability,_ = tank._calc_stability(init_idx)
            last_stability,_ = tank._calc_stability(final_idx)

            #f_errors[i,f_] = f.sdest_record[-1]
            f_errors[i,f_] = f_error
            f_initial[i,f_] = f_error_0

            end_costs[i,f_] = last_cost
            init_costs[i,f_] = initial_cost

            #import pdb;pdb.set_trace()

            end_lin[i,f_] = last_linearity
            init_lin[i,f_] = initial_linearity

            end_stab[i,f_] = last_stability
            init_stab[i,f_] = initial_stability
            #f_initial[i,f_] = (f.size - f.est_record[0])**2 / f.est_record[0]
    out_list = [f_errors,f_initial,
                end_costs,init_costs,    
                end_lin,init_lin,
                end_stab,init_stab]
    return out_list

## should I do the whole range? 

s_res = 11
l_res = s_res
a_res = s_res

s_list = np.linspace(0,1,s_res)
l_list = np.linspace(-1,1,l_res)
a_list = np.linspace(0,1,a_res)
c_list = np.linspace(0,1,a_res)

np.set_printoptions(formatter={'all':lambda x: str(x)})
shifted_l = (l_list + 1)/2
l_labels = np.round(np.tan(np.array(np.pi/2 - shifted_l*np.pi/2)),1).astype('str')
l_labels[0] = 'inf' 

a_labels = np.round(np.tan(np.array(a_list)*np.pi/2) * 20,1).astype('str')
a_labels[-1] = 'inf'

params = Params()
params.n_rounds = 10 
params.n_iterations = 3

def run_many_sims(s,params):
    params = copy.deepcopy(params)
    some_errors = np.empty([l_res,a_res,a_res,16])

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
                sim_output = run_sim(params)
                f_errors,f_initial = sim_output[:2]
                end_costs,init_costs = sim_output[2:4]
                end_lin,init_lin = sim_output[4:6]
                end_stab,init_stab = sim_output[6:8]

                #f_errors,f_initial,end_costs,init_costs = run_sim(params)
                some_errors[l_,a_,c_,:2] = mean_sem(f_errors)
                some_errors[l_,a_,c_,2:4] = mean_sem(f_initial)
                some_errors[l_,a_,c_,4:6] = mean_sem(end_costs)
                some_errors[l_,a_,c_,6:8] = mean_sem(init_costs)
                some_errors[l_,a_,c_,8:10] = mean_sem(end_lin)
                some_errors[l_,a_,c_,10:12] = mean_sem(init_lin)
                some_errors[l_,a_,c_,12:14] = mean_sem(end_stab)
                some_errors[l_,a_,c_,14:16] = mean_sem(init_stab)
    return some_errors

print('Running bayesian simulation')
if False: ## Handy for debug
    all_results = np.empty([s_res,l_res,a_res,a_res,16])
    for s_ in tqdm(range(s_res)):
        s = s_list[s_]
        all_results[s_] = run_many_sims(s,params)

else:
    all_results = Parallel(n_jobs=11)(delayed(run_many_sims)(s,params) for s in s_list)
    all_results = np.array(all_results)

print('Running no-update simulation')
null_params = params.copy()
null_params.update_method = None
if False: ## Handy for debug
    null_results = np.empty([s_res,l_res,a_res,a_res,16])
    for s_ in tqdm(range(s_res)):
        s = s_list[s_]
        null_results[s_] = run_many_sims(s,null_params)

else:
    null_results = Parallel(n_jobs=11)(delayed(run_many_sims)(s,null_params) for s in s_list)
    null_results = np.array(null_results)

def parse_results(all_results):
    all_errors = all_results[:,:,:,:,0:4]
    all_costs = all_results[:,:,:,:,4:8]
    all_lins = all_results[:,:,:,:,8:12]
    all_stabs = all_results[:,:,:,:,12:16]

    mean_errors = all_errors[:,:,:,:,0]
    sem_errors = all_errors[:,:,:,:,1]
    mean_init = all_errors[:,:,:,:,2]
#sem_init = all_errors[:,:,:,:,3]

    mean_costs = all_costs[:,:,:,:,0]
    sem_costs = all_costs[:,:,:,:,1]
    init_costs = all_costs[:,:,:,:,2]

    mean_lins = all_lins[:,:,:,:,0]
    sem_lins = all_lins[:,:,:,:,1]
    init_lins = all_lins[:,:,:,:,2]

    mean_stabs = all_stabs[:,:,:,:,0]
    sem_stabs = all_stabs[:,:,:,:,1]
    init_stabs = all_stabs[:,:,:,:,2]
    return [mean_errors,sem_errors,mean_init],[mean_costs,sem_costs,init_costs],[mean_lins,sem_lins,init_lins],[mean_stabs,sem_stabs,init_stabs]


def build_inputs(mean_array,sem_array,init_array):
    s_mean = mean_array[:,1,5,1]
    s_sem = sem_array[:,1,5,1]
    s_init = init_array[:,1,5,1]
    s_data = [s_mean,s_sem,s_init]

    l_mean = mean_array[7,:,5,1]
    l_sem = sem_array[7,:,5,1]
    l_init = init_array[7,:,5,1]
    l_data = [l_mean,l_sem,l_init]

    a_mean = mean_array[7,1,:,1]
    a_sem = sem_array[7,1,:,1]
    a_init = init_array[7,1,:,1]
    a_data = [a_mean,a_sem,a_init]

    c_mean = mean_array[7,1,5,:]
    c_sem = sem_array[7,1,5,:]
    c_init = init_array[7,1,5,:]
    c_data = [c_mean,c_sem,c_init]
    return s_data,l_data,a_data,c_data

def make_plots(s_data,l_data,a_data,c_data,ylabel='None',fill_color='gray',l_style=None,l_label=None,fax=None):
    if fax is None:
        fig,axes = plt.subplots(1,4,sharey=True)
    else:
        fig,axes = fax
    s_mean,s_sem,s_init = s_data
    l_mean,l_sem,l_init = l_data
    a_mean,a_sem,a_init = a_data
    c_mean,c_sem,c_init = c_data

    axes[0].plot(s_list,s_mean,color='black',linestyle=l_style)
    axes[0].plot(s_list,s_init,linestyle=':',color=fill_color)

    axes[0].fill_between(s_list,s_mean-s_sem,s_mean+s_sem,alpha=0.5,color=fill_color)
    axes[0].set_xlabel('s value')
    axes[0].set_ylabel(ylabel)
    axes[0].axvline(0.7,color='red',linestyle=':')

    axes[1].plot(l_list,l_mean,color='black',linestyle=l_style)
    axes[1].plot(l_list,l_init,linestyle=':',color=fill_color)
    axes[1].fill_between(l_list,l_mean-l_sem,l_mean+l_sem,alpha=0.5,color=fill_color)
    axes[1].set_xlabel('l value')
    axes[1].axvline(-0.8,color='red',linestyle=':')

    axes[1].set_xticks(l_list)
    axes[1].set_xticklabels(l_labels,rotation=45)
    axes[1].invert_xaxis()

    axes[2].plot(a_list,a_mean,color='black',linestyle=l_style)
    axes[2].plot(a_list,a_init,linestyle=':',color=fill_color)
    axes[2].fill_between(a_list,a_mean-a_sem,a_mean+a_sem,alpha=0.5,color=fill_color)
    axes[2].axvline(0.5,color='red',linestyle=':')

    axes[2].set_xlabel('self assessment error')
    axes[2].set_xticklabels(a_labels,rotation=45)
    axes[2].set_xticks(a_labels)

    axes[3].plot(c_list,c_mean,color='black',linestyle=l_style,label=l_label)
    axes[3].plot(c_list,c_init,linestyle=':',color=fill_color)
    axes[3].fill_between(c_list,c_mean-c_sem,c_mean+c_sem,alpha=0.5,color=fill_color)
    axes[3].axvline(0.1,color='red',linestyle=':',label='default value')

    axes[3].set_xlabel('opp assessment error')
    axes[3].set_xticks(a_labels)
    axes[3].set_xticklabels(a_labels,rotation=45)
    return fig,axes

error_info,cost_info,lin_info,stab_info = parse_results(all_results)
null_error_info,null_cost_info,null_lin_info,null_stab_info = parse_results(null_results)

error_inputs = build_inputs(*error_info)
cost_inputs = build_inputs(*cost_info)
lin_inputs = build_inputs(*lin_info)
stab_inputs = build_inputs(*stab_info)

null_error_inputs = build_inputs(*null_error_info)
null_cost_inputs = build_inputs(*null_cost_info)
null_lin_inputs = build_inputs(*null_lin_info)
null_stab_inputs = build_inputs(*null_stab_info)


#error_inputs = build_inputs(mean_errors,sem_errors,mean_init)
#cost_inputs = build_inputs(mean_costs,sem_costs,init_costs)
#lin_inputs = build_inputs(mean_lins,sem_lins,init_lins)
#stab_inputs = build_inputs(mean_stabs,sem_stabs,init_stabs)

fig1,axes1 = make_plots(*null_error_inputs,fill_color='gray',l_style=None,ylabel='Estimate Error')

fig2,axes2 = make_plots(*null_cost_inputs,fill_color='gray',l_style=None,ylabel='Mean Contest Cost')
fig3,axes3 = make_plots(*null_lin_inputs,fill_color='gray',l_style=None,ylabel='Linearity')
fig4,axes4 = make_plots(*null_stab_inputs,fill_color='gray',l_style=None,ylabel='Stability')


fig1,axes1 = make_plots(*error_inputs,fax=(fig1,axes1),ylabel='Estimate Error',fill_color='tab:blue',l_style='dashdot',l_label='Bayes Updating')
fig2,axes2 = make_plots(*cost_inputs,fax=(fig2,axes2),ylabel='Mean Contest Cost',fill_color='tab:blue',l_style='dashdot',l_label='Bayes Updating')
fig3,axes3 = make_plots(*lin_inputs,fax=(fig3,axes3),ylabel='Linearity',fill_color='tab:blue',l_style='dashdot',l_label='Bayes Updating')
fig4,axes4 = make_plots(*stab_inputs,fax=(fig4,axes4),ylabel='Stability',fill_color='tab:blue',l_style='dashdot',l_label='Bayes Updating')

fig1.legend()
fig2.legend()
fig3.legend()
fig4.legend()

plt.show()
