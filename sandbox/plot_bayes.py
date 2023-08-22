## Load required packages
import numpy as np

import itertools,random

from scipy.special import rel_entr
from scipy.stats import mode
from scipy.stats import norm

from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib import cm


def win_by_ratio(r, k=0.1,m=0):
    # sigmoid function
    # use k to adjust the slope
    p = 1 / (1 + np.exp(-(r-m) / k)) 
    return p

def likelihood_function_size(x,x_opp=50):
    if x >=x_opp:
        r_diff = (x - x_opp)/x # Will be positive
    elif x_opp > x:
        r_diff = (x - x_opp)/x_opp # Will be negative
    p_win = win_by_ratio(r_diff)
    return p_win
    
def define_likelihood(x_opp=50,xs=np.arange(5,150),win=True):
    likelihood = np.zeros(len(xs))
    if win:
        for s in range(len(xs)):
            likelihood[s] = likelihood_function_size(xs[s],x_opp)
    elif not win:
        for s in range(len(xs)):
            likelihood[s] = 1-likelihood_function_size(xs[s],x_opp)
    return likelihood

def prior_size(t,xs = np.arange(5,150)):
    s_mean = growth_func(t)
    sd = s_mean / 5
    return norm.pdf(xs,s_mean,sd)

def growth_func(t,s_max = 100):
    size = 7 + s_max - s_max/np.exp(t/100)
    return size

## xs need to be the same for prior, likelihood, and post
def update(prior,likelihood,xs = np.arange(5,150)):
    post = prior * likelihood
    post = post / auc(xs,post)
    return post

xs = np.arange(5,150)
t = 100
x_opp = 60
win = True
prior = prior_size(t,xs)

likelihood = define_likelihood(x_opp,win=win)
posterior = update(prior,likelihood)
posterior_i = update(posterior,likelihood)


likelihood_loss = define_likelihood(x_opp,win=False)
posterior_loss = update(prior,likelihood_loss)
posterior_l = update(posterior_loss,likelihood_loss)

fig,ax = plt.subplots()
fig1,ax1 = plt.subplots()
ax.plot(xs,prior,label='Prior')
ax.plot(xs,posterior,label='Posterior (after win)',color='tab:green')
ax.plot(xs,posterior_i,label='Repeated wins',color='gray',alpha=.1)

ax1.plot(xs,prior,label='Prior')
ax1.plot(xs,posterior_loss,label='Posterior (after loss)',color='tab:green')
ax1.plot(xs,posterior_l,color='gray',alpha=.1,label='Repeated losses')

for i in range(100):
    posterior_i = update(posterior_i,likelihood)
    posterior_l = update(posterior_l,likelihood_loss)
    ax.plot(xs,posterior_i,color='gray',alpha=.1)
    ax1.plot(xs,posterior_l,color='gray',alpha=.1)
    
ax.axvline(x_opp,color='black',linestyle=':',label='Opponent Size')
ax1.axvline(x_opp,color='black',linestyle=':',label='Opponent Size')

ax.legend()
fig.show()

ax1.legend()
fig1.show()

fig2,ax2 = plt.subplots()

ax2.plot(xs,likelihood,label='Likelihood',color='tab:orange')
ax2.axvline(x_opp,color='black',linestyle=':',label='Opponent Size')

ax2.legend()
fig2.show()

fig3,ax3 = plt.subplots()
ax3.plot(xs,prior,label='Prior Self Assessment')
ax3.axvline(x_opp,color='black',linestyle=':',label='Opponent Size')
ax3.set_ylim(-0.001,0.051)
ax3.legend()
fig.show()

plt.show()
