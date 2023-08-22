#! /usr/bin/env python

## Script for exploring parameters for path dependency

import numpy as np
from tqdm import tqdm

from fish import Fish
from fight import Fight
from params import Params



def build_fish(params = Params()):
#params = Params()
    params.age = 51
    params.size = 50
    params.prior = True
    params.acuity = 0
    params.awareness = 0.5
    params.set_params()

    f1 = Fish(1,params)
    return f1,params

def build_opp(opp_params = Params(),o_eff = None):
    if o_eff is not None:
        opp_params.effort_method = o_eff
        opp_params.baseline_effort = o_eff[1]
    f_opp = Fish(0,opp_params)
    return f_opp,opp_params

outcomes = [[0,1],[1,0]]


scale = 21
p_space = np.linspace(-1,1,scale)

p_array = np.zeros([scale,scale,scale])
_,params = build_fish()
f_opp,opp_params = build_opp(params.copy(),o_eff = [None,0.3]) ## for now, opponent is fixed

DEBUG = False
for s_ in tqdm(range(scale)):
    s = p_space[s_]
    for f_ in range(scale):
        f = p_space[f_]
        for l_ in range(scale):
            l = p_space[l_]
            post_estimates = [0,0]

            if DEBUG:
                #s,f,l = [0,0,-0.9]
                pass
            params.outcome_params = [s,f,l]
            params.set_params()
            f_opp,opp_params = build_opp(params.copy(),o_eff = None) ## flexible opponent
            for o_ in range(len(outcomes)):
                r1_outcome,r2_outcome = outcomes[o_]

                f1,params = build_fish(params)
                if DEBUG:
                    print('#############')
                    print(params.outcome_params)
                    print(r1_outcome,r2_outcome)
                    print(f1.estimate)
                fight1 = Fight(f_opp,f1,params,outcome=r1_outcome)
                outcome1 = fight1.run_outcome()
                f1.update(outcome1,fight1)
                if DEBUG:
                    print(f1.estimate,f1.effort,f_opp.effort)

                fight2 = Fight(f_opp,f1,params,outcome=r2_outcome)
                outcome2 = fight2.run_outcome()
                f1.update(outcome2,fight2)
                if DEBUG:
                    print(f1.estimate,f1.effort,f_opp.effort)
                post_estimates[o_] = f1.estimate

            e_diff = post_estimates[0] - post_estimates[1]
            p_array[s_,f_,l_] = e_diff
#import pdb; pdb.set_trace()
#break

np.save('./results/recency_array.npy',p_array)
