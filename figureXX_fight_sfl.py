#! /usr/bin/env python

## Script to iteration through the s,f,l, parameter space 
## showing who would win in a fight between two fish

from params import Params
from fish import Fish
from fight import Fight

## there's a transform here, so for the exponentional
## -1 => 0
## 0 => 1
## 1 == infinity  

fight_params = Params()
fight_params.outcome_params = [1,0,-1]
fight_params.set_params()

f1_params = fight_params.copy()
f2_params = fight_params.copy()

f1_params.size = 60
f1_params.effort_method = [None,0.3]
f1_params.baseline_effort = 0.3

f2_params.size = 50
f2_params.effort_method = [None,0.6]
f2_params.baseline_effort = 0.6

f1 = Fish(1,f1_params)
f2 = Fish(2,f2_params)

fight = Fight(f1,f2,fight_params)
outcome = fight.run_outcome()

print(outcome,f1.effort,f2.effort)
print(fight.winner.idx)
print(fight.p_win)
print(f1.params.L,f1.params.F,f1.params.S)
