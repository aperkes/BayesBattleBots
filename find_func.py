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

def art(x,a=1):

    K_ = (1 + np.exp(k*(m-1)))
    
    X = k * (m-x**a)
    return K_ / (1 + np.exp(X)) / 2

def bob(x,B=2,r=2):
    y = (x**r)**B / ((x**r)**B + (1-x**r)**B)
    return y

def hugh(x,p=0.35,d=6):
    H = 1/(1 + (p/(1-p))**d / (x/(1-x))**d)

from scipy import stats

def cory(x,s=50):
## Here, it's the CDF of being bigger, for some std
    x_abs = x*50 - 50
    var = 10
    y = 1 - stats.norm.cdf(0,x_abs,10)
    return y


m = 0.5
k= 10
a = 2

S = 1 / (1 + np.exp(k*m))
K = 1 / (1/(1 + np.exp(-k*(1-m))) - S)
K_ = (1 + np.exp(k*(m-1)))

#funk = F._wager_curve_sig
funk = cory
#print(foo(0),foo(m),foo(1))
print(funk(0),funk(m),funk(1))

#xs = np.linspace(-3,3,100)
xs = np.linspace(0,1,100)
ys = funk(xs)
for a in [0,0.25,0.5,1,2,4,np.inf]:
    F.params.a = a
    ys = funk(xs,a)

    plt.plot(xs,ys,label='a= ' + str(a))

plt.legend()
#plt.plot(xs,ys)
#plt.axvline(m)
#plt.show()
