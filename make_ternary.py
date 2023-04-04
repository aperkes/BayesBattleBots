#! /usr/bin/env python

## Script to make a ternary plot from a .npy array of values

## You'll need to know the correct parameter values for the numpy array

import numpy as np
import pandas as pd
#import plotly.express as px
import ternary
from matplotlib import pyplot as plt

## Assumes all dimensions have the same p_list, 
##  which should be a safe assumption for ternary plots
def np_to_pd(a,p_list,columns = ['s','f','l','output']):
    df_list = []
    for s_ in range(a.shape[0]):
        for f_ in range(a.shape[1]):
            for l_ in range(a.shape[2]):
                df_list.append([p_list[s_],p_list[f_],p_list[l_],a[s_,f_,l_]])
    df = pd.DataFrame(df_list,columns = columns)
    return df

def np_to_dict(a,p_list,keys=['s','f','l']):
    p_dict = {}
    p_list = (np.array(p_list) * 10) + 10 
    p_list = p_list.astype(int)
    for s_ in range(a.shape[0]):
        for f_ in range(a.shape[1]):
            for l_ in range(a.shape[2]):
                #key = (p_list[s_],p_list[f_],p_list[l_])
                key = (s_,f_,l_)
                p_dict[key] = a[s_,f_,l_]
    return p_dict

def plot_from_dict(p_dict):
    fig,tax = ternary.figure(scale=10)
    tax.heatmap(p_dict,cmap=None)
    return fig,tax 

if __name__ == '__main__':
    a = np.load('./results/lin_tern_0.npy')
    if False: ## Helpful for visualization
        a[:,2,:] = 0
        a[:,3,:] = 0
        a[4,:,:] = 0
        a[5,:,:] = 0
    p_list = [-1,-0.9,-0.8,-0.2,-0.1,0,0.1,0.2,0.8,0.9,1]
    p_dict = np_to_dict(a,p_list)
    fig,tax = plot_from_dict(p_dict)
    tax.bottom_axis_label("s")
    tax.right_axis_label("f")
    tax.left_axis_label("l")
    tax.ticks(axis='brl')
    tax.show()


