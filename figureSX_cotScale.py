#! /usr/bin/env python

## function to plot the cotangent scaling function I use all over the place


import numpy as np
from matplotlib import pyplot as plt

ps = [-1,-0.9,-0.8,-0.5,0,0.5,0.8,0.9,1]
ps = ps[::-1]
cmap = plt.cm.get_cmap('viridis')

xs = np.linspace(0,1,100)

fig,ax = plt.subplots()
for p_ in range(len(ps)):
    p = ps[p_]
    shifted_p = (p + 1) / 2
    P = np.tan(np.pi/2 - shifted_p*np.pi/2)

    ax.plot(xs,xs**P,color=cmap(p_/len(ps)),label=str(p))

ax.set_xlabel('Invested input (Rel Size, Effort, etc)')
ax.set_ylabel('Actual output')

fig.legend()

if True:
    fig.savefig('./figures/figureSX_cotScale.png',dpi=300)
if True:
    plt.show()
