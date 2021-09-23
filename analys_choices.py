
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:24 2017

@author: sarah
"""

import numpy as np
from misc import *
import world
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
import itertools
import matplotlib.pylab as plt
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import os
import pandas as pd
import gc
import pickle
np.set_printoptions(threshold = 100000, precision = 5)
plt.style.use('seaborn-whitegrid')


# always pass a list of classes   
def save_data(file_name, objects):

    with open(file_name, 'wb') as output_file:
        pickle.dump(objects, output_file)

def load_data(file_name):
    
    with open(file_name, 'rb') as file:
        objects = pickle.load(file)

    return objects

    
def extract_object(obj):

    keys = []
    obj_dict = obj.__dict__

    for key in obj_dict:
        keys.append(key)


    return keys, obj_dict


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

trials = 100 #number of trials
T = 5 #number of time steps in each trial
Lx = 4 #grid length
Ly = 5
no = Lx*Ly #number of observations
ns = Lx*Ly #number of states
na = 3 #number of actions
npi = na**(T-1)
nr = 2
nc = ns
actions = np.array([[0,-1], [1,0], [0,1]])
g1 = 14
g2 = 10
start = 2

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''




context = False
obs_unc = False
state_unc = False
repetitions = 5
hidden_states = np.zeros([repetitions*trials, T], dtype="int32")
for i in range(repetitions):
    dir = os.getcwd()    
    ttl = list(dir + "\\dir_gridworld_over-actions_h1_rep_0")
    
    ttl[-1] = str(i)
    objects = load_data("".join(ttl))

    keys, w = extract_object(objects[0])
    environment = w['environment']
    hidden_states[i*trials:(i+1)*trials,:] =  environment.hidden_states
    # agent = w['agent']

"""
plot and evaluate results
"""

vals = np.array([1., 2/3., 1/2., 1./2.])
cert_arr = np.zeros(ns)
for s in range(ns):
    x = s//Ly
    y = s%Ly

    #state uncertainty condition
    if state_unc:
        if (x==0) or (y==3):
            c = vals[0]
        elif (x==1) or (y==2):
            c = vals[1]
        elif (x==2) or (y==1):
            c = vals[2]
        else:
            c = vals[3]

        condition = 'state'

    else:
        c = 1.

    cert_arr[s] = c


#find successful and unsuccessful runs
#goal = np.argmax(utility)
successfull_g1 = np.where(hidden_states[:,-1]==g1)[0]
if context:
    successfull_g2 = np.where(hidden_states[:,-1]==g2)[0]
    unsuccessfull1 = np.where(hidden_states[:,-1]!=g1)[0]
    unsuccessfull2 = np.where(hidden_states[:,-1]!=g2)[0]
    unsuccessfull = np.intersect1d(unsuccessfull1, unsuccessfull2)
else:
    unsuccessfull = np.where(hidden_states[:,-1]!=g1)[0]

#total  = len(successfull)

#plot start and goal state
start_goal = np.zeros((Lx,Ly))

x_y_start = (start//Ly, start%Ly)
start_goal[x_y_start] = 1.
x_y_g1 = (g1//Ly, g1%Ly)
start_goal[x_y_g1] = -1.
x_y_g2 = (g2//Ly, g2%Ly)
start_goal[x_y_g2] = -2.

palette = [(159/255, 188/255, 147/255),
            (135/255, 170/255, 222/255),
            (242/255, 241/255, 241/255),
            (242/255, 241/255, 241/255),
            (199/255, 174/255, 147/255),
            (199/255, 174/255, 147/255)]

#set up figure params
factor = 3
grid_plot_kwargs = {'vmin': -2, 'vmax': 2, 'center': 0, 'linecolor': '#D3D3D3',
                    'linewidths': 7, 'alpha': 1, 'xticklabels': False,
                    'yticklabels': False, 'cbar': False,
                    'cmap': palette}#sns.diverging_palette(120, 45, as_cmap=True)} #"RdBu_r",

# plot grid
fig = plt.figure(figsize=[factor*5,factor*4])

ax = fig.gca()

annot = np.zeros((Lx,Ly))
for i in range(Lx):
    for j in range(Ly):
        annot[i,j] = i*Ly+j

u = sns.heatmap(start_goal, ax = ax, **grid_plot_kwargs, annot=annot, annot_kws={"fontsize": 40})
ax.invert_yaxis()
plt.savefig('grid.svg', dpi=600)
#plt.show()

# set up paths figure
fig = plt.figure(figsize=[factor*5,factor*4])

ax = fig.gca()

u = sns.heatmap(start_goal, zorder=2, ax = ax, **grid_plot_kwargs)
ax.invert_yaxis()

#find paths and count them
n1 = np.zeros((ns, na))

for i in successfull_g1:

    for j in range(T-1):
        d = hidden_states[i, j+1] - hidden_states[i, j]
        if d not in [1,-1,Ly,-Ly,0]:
            print("ERROR: beaming")
        if d == 1:
            n1[hidden_states[i, j],0] +=1
        if d == -1:
            n1[hidden_states[i, j]-1,0] +=1
        if d == Ly:
            n1[hidden_states[i, j],1] +=1
        if d == -Ly:
            n1[hidden_states[i, j]-Ly,1] +=1

n2 = np.zeros((ns, na))

if context:
    for i in successfull_g2:

        for j in range(T-1):
            d = hidden_states[i, j+1] - hidden_states[i, j]
            if d not in [1,-1,Ly,-Ly,0]:
                print("ERROR: beaming")
            if d == 1:
                n2[hidden_states[i, j],0] +=1
            if d == -1:
                n2[hidden_states[i, j]-1,0] +=1
            if d == Ly:
                n2[hidden_states[i, j],1] +=1
            if d == -Ly:
                n2[hidden_states[i, j]-Ly,1] +=1

un = np.zeros((ns, na))

for i in unsuccessfull:

    for j in range(T-1):
        d = hidden_states[i, j+1] - hidden_states[i, j]
        if d not in [1,-1,Ly,-Ly,0]:
            print("ERROR: beaming")
        if d == 1:
            un[hidden_states[i, j],0] +=1
        if d == -1:
            un[hidden_states[i, j]-1,0] +=1
        if d == Ly:
            un[hidden_states[i, j],1] +=1
        if d == -Ly:
            un[hidden_states[i, j]-4,1] +=1

total_num = n1.sum() + n2.sum() + un.sum()

if np.any(n1 > 0):
    n1 /= total_num

if np.any(n2 > 0):
    n2 /= total_num

if np.any(un > 0):
    un /= total_num

#plotting
for i in range(ns):

    x = [i%Ly + .5]
    y = [i//Ly + .5]

    #plot uncertainties
    if obs_unc:
        plt.plot(x,y, 'o', color=(219/256,122/256,147/256), markersize=factor*12/(agent.perception.generative_model_observations[i,i])**2, alpha=1.)
    if state_unc:
        plt.plot(x,y, 'o', color=(100/256,149/256,237/256), markersize=factor*12/(cert_arr[i])**2, alpha=1.)

    #plot unsuccessful paths
    for j in range(2):

        if un[i,j]>0.0:
            if j == 0:
                xp = x + [x[0] + 1]
                yp = y + [y[0] + 0]
            if j == 1:
                xp = x + [x[0] + 0]
                yp = y + [y[0] + 1]

            plt.plot(xp,yp, '-', color='#D5647C', linewidth=factor*75*un[i,j],
                        zorder = 9, alpha=1)

#set plot title
#plt.title("Planning: successful "+str(round(100*total/trials))+"%", fontsize=factor*9)

#plot successful paths on top
for i in range(ns):

    x = [i%Ly + .5]
    y = [i//Ly + .5]

    for j in range(2):

        if n1[i,j]>0.0:
            if j == 0:
                xp = x + [x[0] + 1]
                yp = y + [y[0]]
            if j == 1:
                xp = x + [x[0] + 0]
                yp = y + [y[0] + 1]
            plt.plot(xp,yp, '-', color='#4682B4', linewidth=factor*75*n1[i,j],
                        zorder = 10, alpha=1)

#plot successful paths on top
if context:
    for i in range(ns):

        x = [i%Ly + .5]
        y = [i//Ly + .5]

        for j in range(2):

            if n2[i,j]>0.0:
                if j == 0:
                    xp = x + [x[0] + 1]
                    yp = y + [y[0]]
                if j == 1:
                    xp = x + [x[0] + 0]
                    yp = y + [y[0] + 1]
                plt.plot(xp,yp, '-', color='#55ab75', linewidth=factor*75*n2[i,j],
                            zorder = 10, alpha=1)


#print("percent won", total/trials, "state prior", np.amax(utility))

plt.show()
plt.savefig('fig2.png')