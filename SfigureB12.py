#from fish import Fish
from fish import Fish
from tank import Tank
from fight import Fight
from params import Params

from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

params = Params()
params.n_rounds = 30

params.size = None
params.mean_size = 50
params.sd_size = 0

## Give all fish the same, centered on 50 prior
params.prior = True
fish0 = Fish(0,params)
base_prior = fish0.prior
params.prior = fish0.prior


params.set_params()
fishes = []
sizes = [20,35,50,65,80]
#sizes = [50,50,50,50,50]
if False:
    params.r_rhp = 1
    params.energy_cost = True
    food = True
else:
    food = False

#fishes = [Fish(f,params) for f in range(5)]
params.size = 50

iterations = 100
#iterations = 3
s_res = 11
s_list = np.linspace(0,1,s_res)

null_params = params.copy()
null_params.update_method = None

def run_one_sim(s,params):
    params = params.copy()
    some_results = np.empty([iterations,2])
    params.outcome_params[0] = s
    params.set_params()

    for i in tqdm(range(iterations)):
        fishes = []
        for f_ in range(5):
            params.size = sizes[f_]
            fishes.append(Fish(f_,params))
        tank = Tank(fishes,params)
        tank.run_all(progress = False)


        stabs = []
        lins = []

        init_idx = slice(0,1)
        final_idx = slice(-3,None)

        initial_linearity,_ = tank._calc_linearity(init_idx)
        last_linearity,_ = tank._calc_linearity(final_idx)
        some_results[i] = initial_linearity,last_linearity
    return some_results

if False:
    null_array = np.empty([s_res,iterations,2])

    results_array = np.empty([s_res,iterations,2])
    for s_ in range(s_res):
        s = s_list[s_]
        null_array[s_] = run_one_sim(s,null_params)
        results_array[s_] = run_one_sim(s,params)

else:
    results_array = Parallel(n_jobs=11)(delayed(run_one_sim)(s,params) for s in s_list)
    results_array = np.array(results_array)
    null_array = Parallel(n_jobs=11)(delayed(run_one_sim)(s,null_params) for s in s_list)
    null_array = np.array(null_array)

fig,ax = plt.subplots()

mean_init = np.mean(results_array[:,:,0],axis=1)

mean_final = np.mean(results_array[:,:,1],axis=1)
sem_final = np.std(results_array[:,:,1],axis=1) / np.sqrt(iterations)
null_init = np.mean(null_array[:,:,0],axis=1)

null_final = np.mean(null_array[:,:,1],axis=1)
sem_null = np.std(null_array[:,:,1],axis=1) / np.sqrt(iterations)

ax.plot(s_list,mean_final,color='black',linestyle='dashdot',label='Final Bayes')
ax.plot(s_list,mean_init,color='darkblue',linestyle='dashdot',label='First-round Bayes')
ax.fill_between(s_list,mean_final - sem_final,mean_final + sem_final,color='darkblue',alpha=0.5)

ax.plot(s_list,null_init,color='black',linestyle=':',label='first-round no-update')
ax.plot(s_list,null_final,color='black',label='final no-update')
ax.fill_between(s_list,null_final - sem_null,null_final + sem_null,color='gray',alpha=0.5)

ax.set_ylim([0.5,1])
ax.set_ylabel('Linearity')
ax.set_xlabel('s value')

ax.legend()
plt.show()

