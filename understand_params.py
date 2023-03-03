
import numpy as np
from matplotlib import pyplot as plt

from fish import Fish
from fight import Fight
from params import Params

params = Params()
f1 = Fish(1,params)
f2 = Fish(2,params)
c = Fight(f1,f2)

f2_size = 47
f2_effort = 0.37

s = 0.9
f = 0.1
l = 0.1

L = np.tan(np.pi/2 - l*np.pi/2)
c.params.L = L

xs = np.linspace(1,100,100)

fig,ax= plt.subplots()

ax.axvline(f2_size)
for effort in [0.1,0.3,0.5,1]:
    for x in xs:
        if x <= f2_size:
            f1_wager = (x/f2_size)**s * effort**f
            f2_wager = f2_effort**f
        else:
            f1_wager = effort**f
            f2_wager = (f2_size/x)**s * f2_effort**f

        min_normed = min([f1_wager,f2_wager])/max([f1_wager,f2_wager])
        ax.scatter(x,min_normed,color='red')
        p_win = c._wager_curve(min_normed,l)
        if f1_wager >= f2_wager:
            p_win = 1-p_win
        ax.scatter(x,p_win,color='black')
plt.show() 
