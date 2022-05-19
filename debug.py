## Simple script to run debug so I don't have to keep rebooting python

from fish import Fish
from tank import Tank
from simulation import Simulation 

fishes = [Fish(n) for n in range(5)]
tank = Tank(fishes)
tank.run_all()

s = Simulation()
lin,p = s._calc_linearity(tank)

print('fishes:')

for f in tank.fishes:
    print(f.idx)
print('win record:')
print(tank.win_record)
print(tank.h_matrix)
print(lin,p)
t_stats = s._get_tank_stats(tank)

print(t_stats)
