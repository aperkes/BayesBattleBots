#! /usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

a0 = np.load('./results/lin_tern_0_ac.npy')
a20 = np.load('./results/lin_tern_20_ac.npy')

a0 = a0[:,:]
a20 = a20[:,:]
print(a0[0])


fig,(ax1,ax2,ax3) = plt.subplots(1,3)

ax1.imshow(a0,vmax=1,vmin=0.75)
ax1.set_xlabel('Awareness')
ax1.set_ylabel('Acuity')

ax2.imshow(a20,vmax=1,vmin=0.75)
ax3.imshow(a20-a0,cmap='PiYG',vmax=0.3,vmin=-0.3)

plt.show()
