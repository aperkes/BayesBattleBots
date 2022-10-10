import numpy as np
from matplotlib import pyplot as plt

## This assumes perfect of size, calculate optimal effort
x_size = 60
o_size = 40
o_eff = 1.0 

effort_space = np.linspace(0,1,100)
s,e,l = .7,.3,.01

L = np.tan((np.pi - l)/2)

def calculate_effort(x_eff,o_eff,x_size,o_size):
    if x_size <= o_size:
        p_win = calculate_smaller(x_eff,o_eff,x_size,o_size)
    else:
        p_win = calculate_bigger(x_eff,o_eff,x_size,o_size)
    return p_win

def calculate_smaller(x_eff,o_eff,x_size,o_size):
    #inflection_point = o_eff * ((o_size/x_size)**s)**(1/e)
    x_wager = (x_size/o_size)**s * x_eff**e
    o_wager = o_eff**e
    if x_wager <= o_wager:
    #if x_eff <= inflection_point: ## it'll be the smaller wager
        p_win = (x_wager/o_wager) ** L / 2 
    else:
        p_win = 1 - (o_wager/x_wager) ** L / 2
    return p_win 


def calculate_bigger(x_eff,o_eff,x_size,o_size): 
    #inflection_point = o_eff * ((x_size/o_size)**s)**(1/e)
    x_wager = x_eff**e
    o_wager = o_eff**e * (o_size/x_size)**s
    if x_wager <= o_wager:
    #if x_eff <= inflection_point: ## it'll be the smaller wager
        p_win = (x_wager/o_wager)**L / 2 
    else:
        p_win = 1 - (o_wager/x_wager)**L / 2
    return p_win

def build_pwin_array(o_eff,x_size,o_size):
    pwin_array = np.empty_like(effort_space)
    for f in range(len(effort_space)):
        x_eff = effort_space[f]
        pwin_array[f] = calculate_effort(x_eff,o_eff,x_size,o_size)
    return pwin_array

def plot_effort(pwin_array = None):
    if pwin_array is None:
        pwin_array = build_pwin_array(o_eff,x_size,o_size)
    fig,ax = plt.subplots()
    ax.plot(effort_space,pwin_array,label='P(win)')
    ax.plot(effort_space,effort_space,label='Effort')
    ax.plot(effort_space,pwin_array-effort_space,label='Expected Reward')
    fig.legend()
    return fig,ax

if __name__ == "__main__":
    print(L)
    #fig,ax = plot_effort(pwin_array)
    max_space = np.empty_like(effort_space)
    full_space = np.empty([100,100])
    for x_f in range(len(effort_space)):
        for o_f in range(len(effort_space)): 
            x_eff = effort_space[x_f]
            o_eff = effort_space[o_f]
            #o_eff = 1
            full_space[x_f,o_f] = calculate_effort(x_eff,o_eff,x_size,o_size)
    mean_reward = np.mean(full_space,axis=1) - effort_space
    fig,ax = plt.subplots()
    ax.plot(effort_space,mean_reward)
    fig.show()
    plt.show()
