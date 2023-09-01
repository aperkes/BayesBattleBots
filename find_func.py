import matplotlib.pyplot as plt

import numpy as np
from fight import Fight
from fish import Fish

F = Fight(Fish(),Fish())

## set K at the top of the params, and store it there.
def bar(x):
    X = k * (m-x)
    return K * (1 / (1 + np.exp(X)) - S)

def foo(x):
    X = k * (m-x)
    return K_ / (1 + np.exp(X))

def tanthing(x):
    a = 10
    k = -1 / (2 * np.arctan(-a/(2*np.pi)))
    j = np.arctan(a*(x-m)/np.pi)
    y = k*j + m 
    return y

## My original function
def art(x=0.7,a=1):
    X = k * (m-x**a)
    return K_ / (1 + np.exp(X)) / 2

## forget where this came from
def bob(x,B=2,r=2):
    y = (x**r)**B / ((x**r)**B + (1-x**r)**B)
    return y

## H function from staty on twitter
def hugh(x=0.7,p=0.35,d=6):
    if x == 1:
        H = 0.5
    H = 1/(1 + (p/(1-p))**d / (x/(1-x))**d)
    return H

p = 0.35
d=6
A = (p/(1-p))**-d
## G function from staty on twitter
def greg(x=0.7,p=0.35,d=6):
    return A / (A + (x/(1-x))**-d)

B = (p/(1-p))**d
## F function from staty on twitter
def frank(x=0.8,p=0.35,d=6):
    X = (x/(1-x))**d
    return X/ (X+B)

from scipy import stats

## CDF of difference between two distributions
def cory(x=0.7,s=50):
## Here, it's the CDF of being bigger, for some std
    x_abs = x*50 - 50
    var = 1*x*50
    y = 1 - stats.norm.cdf(0,x_abs,var)
    return y

from scipy.special import logit
## Logit-sigmoid
def perrin(x=0.7):
    x_ = logit(x)
    return 1 / (1 + np.exp(-x_))

import math
## Hyperbolic tangent tanh
def taco(x=0.7):
    a = 5
    h = 0.7
    #math.tanh(a*(x-h))/2 + 1/2    
    y = np.tanh(a*(x-h))/2 + 1/2
    return y

m = 0.5
k= 10
K_ = (1 + np.exp(k*(m-1)))
a = 2

S = 1 / (1 + np.exp(k*m))
K = 1 / (1/(1 + np.exp(-k*(1-m))) - S)
K_ = (1 + np.exp(k*(m-1)))

import timeit

print('sigmoidal',timeit.Timer(art).timeit(number=10000))
print('H cleverness',timeit.Timer(hugh).timeit(number=10000))
print('G more cleverness',timeit.Timer(greg).timeit(number=10000))
print('F more cleverness',timeit.Timer(frank).timeit(number=10000))
print('logit-sigmoid:',timeit.Timer(perrin).timeit(number=10000))
print('cdf',timeit.Timer(cory).timeit(number=10000))
print('tanh',timeit.Timer(taco).timeit(number=10000))
#funk = F._wager_curve_sig
funk = perrin
#print(foo(0),foo(m),foo(1))
print(funk(0.001),funk(m),funk(.99))
print(hugh(0.001),hugh(0.5),hugh(.99))
#xs = np.linspace(-3,3,100)
xs = np.linspace(0.01,1,100)
ys = funk(xs)
if False:
    for a in [0,0.25,0.5,1,2,4,np.inf]:
        F.params.a = a
        ys = funk(xs,a)

        plt.plot(xs,ys,label='a= ' + str(a))

#plt.legend()
plt.plot(xs,ys)
#plt.axvline(m)
plt.show()
