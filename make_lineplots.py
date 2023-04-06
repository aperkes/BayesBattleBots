#! /usr/bin/env python


# script to plot line plots for effort params

import numpy as np
from matplotlib import pyplot as plt

a = np.load('./results/eff_array_ac.npy')
#a = eff_array[:,5,:] ## don't need boldness for now

low_a = a[7]
med_a = a[5]
high_a =a[3]

mean_low = np.mean(low_a,axis=1)
std_low = np.std(low_a,axis=1) / np.sqrt(1000)

mean_med = np.mean(med_a,axis=1)
std_med = np.std(med_a,axis=1) / np.sqrt(1000)

mean_high = np.mean(high_a,axis=1)
std_high = np.std(high_a,axis=1) / np.sqrt(1000)

fig,ax  = plt.subplots()

xs = np.linspace(0,1,11)
ax.plot(xs,mean_low,color='black')
ax.fill_between(xs,mean_low - std_low,mean_low + std_low,alpha=0.5,color='red')

ax.plot(xs,mean_med,color='black')
ax.fill_between(xs,mean_med - std_med,mean_med + std_med,alpha=0.5,color='blue')

ax.plot(xs,mean_high,color='black')
ax.fill_between(xs,mean_high - std_high,mean_high + std_high,alpha=0.5,color='green')

fig.savefig('./figures/line_plotAC.png',dpi=300)

