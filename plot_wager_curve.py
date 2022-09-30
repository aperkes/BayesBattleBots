import numpy as np
from scipy.special import logit
from matplotlib import pyplot as plt

def wager_curve(w,l=.25):
    a = logit(1-l)
    prob_win = w ** (float(np.abs(a))**np.sign(a)) / 2
    return prob_win

fig,ax = plt.subplots()

xs = np.linspace(0,1,50)

ax.plot(xs,wager_curve(xs,l=.0000001))
ax.plot(xs,wager_curve(xs,l=.0001))
ax.plot(xs,wager_curve(xs,l=.01))
ax.plot(xs,wager_curve(xs,l=.5))
ax.plot(xs,wager_curve(xs,l=.99))
ax.plot(xs,wager_curve(xs,l=.999))

plt.show()
