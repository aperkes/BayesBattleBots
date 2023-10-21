#from fish import Fish
if False:
    from OLDfish import Fish
    from OLDtank import Tank
    from OLDfight import Fight
    from OLDparams import Params
else:
    from fish import Fish
    from tank import Tank
    from fight import Fight
    from params import Params

from matplotlib import pyplot as plt

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

import numpy as np

print(params.outcome_params)
params.set_params()
print(params.outcome_params)
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
iterations = 3
s_res = 11
s_list = np.linspace(0,1,s_res)

results_array = np.empty([s_res,iterations,2])

for s_ in range(s_res):
    s = s_list[s_]
    print(s)
    params.outcome_params[0] = s
    params.set_params()

    for i in range(iterations):
        tank = Tank(fishes,params)
        tank.run_all(progress = False)


        stabs = []
        lins = []

        init_idx = slice(0,3)
        final_idx = slice(-3,None)

        initial_linearity,_ = tank._calc_linearity(init_idx)
        last_linearity,_ = tank._calc_linearity(final_idx)
        results_array[s_,i] = initial_linearity,last_linearity


fig,ax = plt.subplots()

ax.plot(np.mean(results_array[:,:,0],axis=1))
plt.show()

