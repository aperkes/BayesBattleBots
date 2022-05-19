## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation 

fishes = [Fish(n) for n in range(7)]
tank = Tank(fishes,n_fights = 100)
tank.run_all()

s = Simulation()
lin,p = s._calc_linearity(tank)

stab = s._calc_stability(tank)

print(stab)

t_stats = s._get_tank_stats(tank)

print(t_stats)
