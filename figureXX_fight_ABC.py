#! /usr/bin/env python

# Script to explore effort parameters

from fish import Fish
from fight import Fight
from params import Params

params = Params(size=50)

f1 = Fish(1,params)
f2 = Fish(2,params)
fight = Fight(f1,f2,params)

fight.run_outcome()
#eff = f1.choose_effort(f2,fight)
print(f1.size,f2.size)
print(f1.guess,f1.estimate,f2.guess,f2.estimate)
print(f1.effort)
