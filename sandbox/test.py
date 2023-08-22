from matplotlib import pyplot as plt
import numpy as np
from fish import Fish


f = Fish(0)

prior = np.array(f.prior)

fig,(ax,ax1) = plt.subplots(2)

for n in range(1,10):
    f.prior = prior ** n
    f.prior = f.prior / sum(f.prior)
    mu,std = f.get_stats()
    ax.plot(f.xs,f.prior)
    ax1.scatter(n,f.prior_std)
    ax1.scatter(n,std)

plt.show()


