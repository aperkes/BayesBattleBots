#! /usr/bin/env python


## Needs to take a dict of estimates or probabilities and make the chaos figure

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons


est_array = np.load('./est_array_debug.npy',allow_pickle=True)
prob_array = np.load('./prob_array_debug.npy',allow_pickle=True)

n_f,n_a = est_array.shape[0]-1,est_array.shape[-1]-1
est_dict = est_array[2,2,2,1,1]
prob_dict = prob_array[2,2,2,1,1]

MAX_THIC = 5
params = [2,2,2,1,1]

def plot_dict(est_dict,prob_dict = None):
    #fig,ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([25,75])
    fig.subplots_adjust(left=0.25,bottom=0.45)

    line_dict = {}
    for k in est_dict.keys():
        for k2 in est_dict.keys():
            if len(k2) == len(k) + 1 and k2[:-1] == k:
                if k2[0] == 'l':
                    cor = 'darkblue'
                else:
                    cor = 'gold'
                if prob_dict is not None:
                    thic = prob_dict[k]
                    if k2[-1] == 'l':
                        thic = 1-thic
                    thic = thic * MAX_THIC
                    thic = np.clip(thic,1,MAX_THIC)
                else:
                    thic = None
                [line_dict[k2]] = ax.plot([len(k),len(k2)],[est_dict[k],est_dict[k2]],linewidth=thic,color=cor,alpha=0.5)

    return fig,ax,line_dict

def sliders_on_changed(val):
    #s,f,l,a,c = params
    s = s_slider.val
    f = f_slider.val
    l = l_slider.val
    a = a_slider.val
    c = c_slider.val

    est_dict = est_array[s,f,l,a,c]
    prob_dict = prob_array[s,f,l,a,c]
    update_lines(est_dict,prob_dict)
    fig.canvas.draw_idle()

def update_lines(est_dict,prob_dict):
    for k in est_dict.keys():
        t0 = k
        for k2 in est_dict.keys():
            if len(k2) == len(k) + 1 and k2[:-1] == k:
                thic = prob_dict[k]
                if k2[-1] == 'l':
                    thic = 1-thic
                thic = thic * MAX_THIC
                thic = np.clip(thic,1,MAX_THIC)

                #thic = prob_dict[k]
                xs = [len(k),len(k2)]
                ys = [est_dict[k],est_dict[k2]]
                line_dict[k2].set_data(xs,ys)
                line_dict[k2].set_linewidth(thic)
    #print(k,k2)
    return 0 ## I know this doesn't do anything, but I feel weird without an ending

fig,ax,line_dict = plot_dict(est_dict,prob_dict)

axis_color = 'lightgoldenrodyellow'

## Add 5 slider bars
s_slider_ax = fig.add_axes([0.25,0.3,0.65,0.03], facecolor=axis_color)
s_slider = Slider(s_slider_ax,'s',0,n_f,valstep=1,valinit=n_f // 2)
s_slider.on_changed(sliders_on_changed)

f_slider_ax = fig.add_axes([0.25,0.25,0.65,0.03], facecolor=axis_color)
f_slider = Slider(f_slider_ax,'f',0,n_f,valstep=1,valinit=n_f // 2)
f_slider.on_changed(sliders_on_changed)

l_slider_ax = fig.add_axes([0.25,0.20,0.65,0.03], facecolor=axis_color)
l_slider = Slider(l_slider_ax,'l',0,n_f,valstep=1,valinit=n_f // 2)
l_slider.on_changed(sliders_on_changed)

a_slider_ax = fig.add_axes([0.25,0.15,0.65,0.03], facecolor=axis_color)
a_slider = Slider(a_slider_ax,'sigma_a',0,n_a,valstep=1,valinit=n_a // 2)
a_slider.on_changed(sliders_on_changed)

c_slider_ax = fig.add_axes([0.25,0.1,0.65,0.03], facecolor=axis_color)
c_slider = Slider(c_slider_ax,'sigma_c',0,n_a,valstep=1,valinit=n_a // 2)
c_slider.on_changed(sliders_on_changed)

## add a reset button
reset_button_ax = fig.add_axes([0.8,0.025,0.1,0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color,hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    s_slider.reset()
    f_slider.reset()
    l_slider.reset()
    a_slider.reset()
    c_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

plt.show()


