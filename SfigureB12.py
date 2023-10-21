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
sizes = [50,50,50,50,50]
if True:
    params.r_rhp = 1
    params.energy_cost = True
    food = True
else:
    food = False
for f_ in range(5):
    params.size = sizes[f_]
    fishes.append(Fish(f_,params))

#fishes = [Fish(f,params) for f in range(5)]
params.size = 50

iterations = 1000
#iterations = 3
s_res = 11
s_list = np.linspace(0,1,s_res)


def run_one_sim(s,params):
    params = params.copy()
    some_results = np.empty([iterations,2])
    params.outcome_params[0] = s
    params.set_params()

    for i in tqdm(range(iterations)):
        tank = Tank(fishes,params)
        tank.run_all(progress = False)


        stabs = []
        lins = []

        init_idx = slice(0,3)
        final_idx = slice(-3,None)

        initial_linearity,_ = tank._calc_linearity(init_idx)
        last_linearity,_ = tank._calc_linearity(final_idx)
        some_results[i] = initial_linearity,last_linearity
    return some_results

if False:
    results_array = np.empty([s_res,iterations,2])
    for s_ in range(s_res):
        s = s_list[s_]
        results_array[s_] = run_one_sim(s)
else:
    results_array = Parallel(n_jobs=11)(delayed(run_one_sim)(s,params) for s in s_list)
    results_array = np.array(results_array)
fig,ax = plt.subplots()

mean_final = np.mean(results_array[:,:,0],axis=1)
sem_final = np.std(results_array[:,:,0],axis=1) / np.sqrt(iterations)
mean_init = np.mean(results_array[:,:,1],axis=1)


ax.plot(s_list,mean_final,color='black')
ax.plot(s_list,mean_init,color='black',linestyle=':')
ax.fill_between(s_list,mean_final - sem_final,mean_final + sem_final,color='darkblue',alpha=0.5)
ax.set_ylim([0.5,1])
ax.set_ylabel('Linearity')
ax.set_xlabel('s value')

plt.show()

