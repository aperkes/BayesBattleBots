#! /usr/bin/env python

## Script for exploring parameters for path dependency

from fish import Fish
from fight import Fight
from params import Params


params = Params()

params.age = 51
params.size = 50
params.prior = True
params.acuity = 0
params.awareness = 0.5
params.set_params()

opp_params = params.copy()
opp_params.effort_method = [None,0.3]
opp_params.baseline_effort = 0.5
f1 = Fish(1,params)
f_opp = Fish(0,opp_params)

print(f1.estimate)
fight1 = Fight(f_opp,f1,params,outcome=0)
#import pdb;pdb.set_trace()
outcome1 = fight1.run_outcome()
f1.update(outcome1,fight1)
print(f1.estimate,f1.effort,f_opp.effort)

fight2 = Fight(f_opp,f1,params,outcome=0)
outcome2 = fight2.run_outcome()
f1.update(outcome2,fight2)
#print(f1.effort)

print(f1.estimate,f1.effort,f_opp.effort)
