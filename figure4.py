#! /usr/bin/env python

## Trying something new:
""" 
This script contains all the code for plotting figure 4 of Ammon's BayesBattleBots paper
For questions, contact Ammon Perkes (perkes.ammon@gmail.com)
"""

import numpy as np
from matplotlib import pyplot as plt
import copy

from fish import Fish
from fight import Fight
from tank import Tank
from simulation import Simulation,SimParams
from matplotlib import cm

## Define some global variables to determine if you will plot and save figures.
PLOT = 1
SAVE = False

params = SimParams()
params.outcome_params = [1,0.1,.1]
params.f_method = 'shuffled'
## Set up a tank
fishes = [Fish(f,prior=True) for f in range(5)]
tank = Tank(fishes,n_fights = 20,death=False,f_method=params.f_method)
tank.run_all()


## Figure 4a: Strength of winner effect

## Figure 4b: Persistance of winner effect

## Figure 4c: Path dependence of winner effect


if PLOT:
    plt.show()
