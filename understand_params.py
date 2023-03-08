
import numpy as np
from matplotlib import pyplot as plt

from fish import Fish
from fight import Fight
from params import Params

params = Params()

s = 0.0
f = 1.0
l = -0.9

params.outcome_params = [s,f,l]
params.set_params()

f1 = Fish(1,params)
f2 = Fish(2,params)
c = Fight(f1,f2)

f2_size = 47
f2_effort = 0.37


#L = np.tan(np.pi/2 - l*np.pi/2)
L = np.tan(np.pi/2 - (l+1)/2*np.pi/2)
S = np.tan(np.pi/2 - (s+1)/2*np.pi/2)
F = np.tan(np.pi/2 - (f+1)/2*np.pi/2)

#c.params.L = L
print(params.outcome_params)
print(params.scaled_params)

xs = np.linspace(1,100,100)

fig,ax= plt.subplots()

ax.axvline(f2_size)
for effort in [0.1,0.3,0.5,1]:
    for x in xs:
        if x <= f2_size:
            f1_wager = (x/f2_size)**S * effort**F
            f2_wager = f2_effort**F
        else:
            f1_wager = effort**F
            f2_wager = (f2_size/x)**S * f2_effort**F

        min_normed = min([f1_wager,f2_wager])/max([f1_wager,f2_wager])
        ax.scatter(x,min_normed,color='red')
        p_win = c._wager_curve(min_normed,l)
        if f1_wager >= f2_wager:
            p_win = 1-p_win
        ax.scatter(x,p_win,color='black')

c.run_outcome()
likelihood = f1._define_likelihood_mut_array(c,win=True)
print(f1.xs,likelihood)
ax.plot(f1.xs,likelihood)
plt.show() 
