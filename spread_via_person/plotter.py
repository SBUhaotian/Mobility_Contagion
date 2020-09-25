import random
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import matplotlib, collections
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16

def __plot_arr(x_arr, ax=None, label="", no_std=False):
    fig = None
    if (ax == None):
        fig, ax = plt.subplots()
    mean_arr = np.mean(x_arr, axis=1)
    #print ("mean_arr:", mean_arr)
    std_arr = np.std(x_arr, axis=1)
    _plot_cases(mean_arr, label=label, ax=ax)
    if (no_std == False):
        ax.fill_between(range(x_arr.shape[0]), mean_arr+std_arr, mean_arr-std_arr, alpha=0.3)
    ax.set_xlabel("# days from start")
    ax.legend(loc='best')
    return fig, ax

def __plot_cdf_infects(x_arr, ax, label):
    mean_arr = np.mean(x_arr, axis=1)*100.0 /100
    Y = np.cumsum(mean_arr)
    ax.plot(range(len(x_arr)), Y, label=label)

def __plot_arr(mean_arr, std_arr, ax=None, label="", no_std=False):
    fig = None
    if (ax == None):
        fig, ax = plt.subplots()
    for i in range(len(mean_arr[0])):
        _plot_cases(mean_arr[:,i], label=label[i], ax=ax)
        if (no_std == False):
            ax.fill_between(range(mean_arr.shape[0]), mean_arr[:,i]+std_arr[:,i], mean_arr[:,i]-std_arr[:,i], alpha=0.3)
        ax.set_xlabel("# days from start")
         
        ax.legend(loc='best')
    _, ymax = ax.get_ylim(); ax.set_ylim([0, ymax])
    return fig, ax

def _plot_cases(cases, ax=None, label=""):
    if (ax == None):
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.plot(cases, label=label)
    ax.set_xlim([0, 30])
    return fig, ax

def saveplotfile(figure, filename, transparent=False, lgd=None, extension="pdf"):
    print ("saving:", filename)
    if (extension == "pdf"):
        if (lgd!=None):
            figure.savefig(filename+'.pdf', format='pdf', dpi=1000, transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            figure.savefig(filename+'.pdf', format='pdf', dpi=1000, transparent=True, bbox_inches='tight')
    elif (extension == "png"):
        if (lgd!=None):
            figure.savefig(filename+'.png', format='png', dpi=600, transparent=transparent, bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            figure.savefig(filename+'.png', format='png', dpi=600, transparent=transparent, bbox_inches='tight')

new_infects = np.array([])
total_infects = np.array([])
active_cases = np.array([])
R_value = np.array([])

new_std = np.array([])
total_std = np.array([])
active_std = np.array([]) 
R_value_std = np.array([])

label = ["x=10\%", "x=20\%", "x=40\%"]
fig_new_infect, ax_new_infect = plt.subplots()
fig_active_case, ax_active_case = plt.subplots()
fig_R, ax_R = plt.subplots()
fig_cdf_infects, ax_cdf_infects = plt.subplots()

ax_new_infect.set_ylabel("# new infections")
ax_active_case.set_ylabel("% active cases")
ax_R.set_ylabel("$R_0$")
ax_cdf_infects.set_ylabel("% total infected"); ax_cdf_infects.set_xlabel("# days from start")

__plot_arr(new_infects, new_std, ax=ax_new_infect, label=label)
__plot_arr(total_infects, total_std, ax = ax_cdf_infects, label=label)
__plot_arr(active_cases, active_std, ax=ax_active_case, label=label)
__plot_arr(R_value, R_value_std, ax=ax_R, label=label)

front_name = "Isolating_"
back_name = "_Ratio"

saveplotfile(fig_new_infect, front_name+"new"+back_name)
saveplotfile(fig_active_case, front_name+"active"+back_name)
saveplotfile(fig_R, front_name+"R"+back_name)
saveplotfile(fig_cdf_infects, front_name+"total"+back_name)



