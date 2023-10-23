#! /usr/bin/env python

# Script to explore effort parameters

import numpy as np

from fish import Fish
from fight import Fight
from params import Params

from tqdm import tqdm

f1_params = Params(size=60)
f2_params = Params(size=40)

iterations = 1000

scale = 11
ac_scale = np.linspace(0,1,scale)
b_scale = np.linspace(-1,1,scale)

eff_array = np.zeros([scale,scale,iterations])

for a_ in tqdm(range(scale)):
    a = ac_scale[a_]
    for c_ in range(scale):
        c = ac_scale[c_]
        f1_params.awareness = a
        f1_params.boldness = 0
        f1_params.acuity = c
        f1_params.set_params()
        for i in range(iterations):
            f1 = Fish(1,f1_params)
            f2 = Fish(2,f2_params)
            fight = Fight(f1,f2,f1_params)
            fight.run_outcome()
            eff_array[a_,c_,i] = f1.effort

np.save('./results/eff_array_ac.npy',eff_array)
