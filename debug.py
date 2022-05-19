## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation 

fishes = [Fish() for n in range(5)]
t = Tank(fishes)
t.run_all()

s = Simulation()
t_stats = s._get_tank_stats(t)

print(t_stats)
