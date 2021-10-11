#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:45:13 2021

@author: terraregina
"""
#%%
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
import os as os
import pandas as pd
import gc
import pickle
np.set_printoptions(threshold = 100000, precision = 5)
plt.style.use('seaborn-whitegrid')

import itertools as itertools
import numpy as np


save = True
data_folder = os.path.join('C:\\Users\\admin\\Desktop\\project\\BalancingControl','data')

const = 0#1e-10

trials = 200 #number of trials
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


# print("start", start)
# print("g2", g2)
# print("g1", g1)
# print("nc", nc)
# print("nr", nr)
# print("npi", npi)
# print("na", na)
# print("ns", ns)
# print("no", no)
# print("trials", trials)
# print("data_folder", data_folder)
# print("save", save)
# print('\n\nrunning simulations\n\n')
# print('-------------------------')

# def run_agent(par_list, trials=trials, T=T, Lx = Lx, Ly = Ly, ns=ns, na=na,var=0.1,run=0,\
#               sample_post = False, sample_other = False, prior_start = True):


def run_agent(par_list, trials=trials, T=T, Lx = Lx, Ly = Ly, ns=ns, na=na,run=0,\
              path=None, print_thoughts = False, print_walks=False):
    #set parameters:
    #obs_unc: observation uncertainty condition
    #state_unc: state transition uncertainty condition
    #goal_pol: evaluate only policies that lead to the goal
    #utility: goal prior, preference p(o)
    # over_actions -> ddm uses prior and likelihood over actions or policies
    obs_unc, state_unc, goal_pol, avg, context, utility = par_list[0]
    over_actions, selector, b, wd, s, A,  = par_list[1:-3]
    sample_post, sample_other, prior_as_start, regime = par_list[-3]
    h,q = par_list[8:]
    # obs_unc, state_unc, goal_pol, selector, context, utility, over_actions, h, q = par_list

    print("q", q)
    print("h", h)
    """
    set action selection method
    """

    if selector == 'dir':
        ac_sel = asl.DirichletSelector(trials = trials, T = T, factor=0.5,
                                      number_of_actions = na, calc_entropy=False, calc_dkl=False, draw_true_post=True)
    elif selector == 'ddm':
        pass
        # sel = 'max'

        # ac_sel = asl.MaxSelector(trials = trials, T = T,
                                    #   number_of_actions = na)

    elif selector == 'rdm':
        ac_sel = asl.RacingDiffusionSelector(trials = trials, T=T,number_of_actions=na,)
    elif selector == 'ardm':        
        ac_sel = asl.AdvantageRacingDiffusionSelector(trials = trials, T=T, number_of_actions=na)
    else:
        print('nothing selected')
    ac_sel.over_actions = over_actions
    ac_sel.sample_other = sample_other
    ac_sel.sample_posterior = sample_post
    ac_sel.prior_as_starting_point = prior_as_start
    ac_sel.A = A
    ac_sel.b = b
    ac_sel.wd = wd
    ac_sel.s = np.sqrt(s)
    if print_walks:
        ac_sel.print_walks = True
    # name_str = selector + '_s'+ str(var)+'_context_' + str(context) + '_over-actions_'+ str(over_actions)+'_h'+str(h) + '_'+str(run)
    """
    create matrices
    """

    vals = np.array([1., 2/3., 1/2., 1./2.])

    #generating probability of observations in each state
    A = np.eye(ns) + const
    np.fill_diagonal(A, 1-(ns-1)*const)

    #state transition generative probability (matrix)
    B = np.zeros((ns, ns, na)) + const

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
        for u in range(na):
            x = s//Ly+actions[u][0]
            y = s%Ly+actions[u][1]

            #check if state goes over boundary
            if x < 0:
                x = 0
            elif x == Lx:
                x = Lx-1

            if y < 0:
                y = 0
            elif y == Ly:
                y = Ly-1

            s_new = Ly*x + y
            if s_new == s:
                B[s, s, u] = 1 - (ns-1)*const
            else:
                B[s, s, u] = 1-c + const
                B[s_new, s, u] = c - (ns-1)*const
                
    B_c = np.broadcast_to(B[:,:,:,np.newaxis], (ns, ns, na, nc))



    """
    create environment (grid world)
    """
    Rho = np.zeros((nr,ns)) + const
    Rho[0,:] = 1 - (nr-1)*const
    Rho[:,np.argmax(utility)] = [0+const, 1-(nr-1)*const]
    util = np.array([1-np.amax(utility), np.amax(utility)])

    environment = env.GridWorld(A, B, Rho, trials = trials, T = T, initial_state=start)

    Rho_agent = np.ones((nr,ns,nc))/ nr


    if True:
        templates = np.ones_like(Rho_agent)
        templates[0] *= 100
        assert ns == nc
        for s in range(ns):
            templates[0,s,s] = 1
            templates[1,s,s] = 100
        dirichlet_rew_params = templates
    else:
        dirichlet_rew_params = np.ones_like(Rho_agent)

    """
    create policies
    """


    pol = np.array(list(itertools.product(list(range(na)), repeat=T-1)))

    #pol = pol[np.where(pol[:,0]>1)]

    npi = pol.shape[0]

    prior_policies = np.ones((npi,nc)) / npi
    dirichlet_pol_param = np.zeros_like(prior_policies) + h

    """
    set state prior (where agent thinks it starts)
    """

    state_prior = np.zeros((ns))

    state_prior[start] = 1

    """
    set context prior and matrix
    """

    context_prior = np.ones(nc)

    trans_matrix_context = np.ones((nc,nc))
    if nc > 1:
        # context_prior[0] = 0.9
        # context_prior[1:] = 0.1 / (nc-1)
        context_prior /= nc
        trans_matrix_context[:] = (1-q) / (nc-1)
        np.fill_diagonal(trans_matrix_context, q)


        
    """
    set up agent
    """
    #bethe agent
    if agent == 'bethe':

        agnt = 'bethe'

        # perception and planning

        bayes_prc = prc.HierarchicalPerception(A, B_c, Rho_agent, trans_matrix_context, state_prior,
                                               util, prior_policies,
                                               dirichlet_pol_params = dirichlet_pol_param,
                                               dirichlet_rew_params = dirichlet_rew_params)

        bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                      trials = trials, T = T,
                      prior_states = state_prior,
                      prior_policies = prior_policies,
                      prior_context = context_prior,
                      number_of_states = ns,
                      learn_habit = True,
                      learn_rew = True,
                      #save_everything = True,
                      number_of_policies = npi,
                      number_of_rewards = nr)
    #MF agent
    else:

        agnt = 'mf'

        # perception and planning

        bayes_prc = prc.MFPerception(A, B, state_prior, utility, T = T)

        bayes_pln = agt.BayesianMFPlanner(bayes_prc, [], ac_sel,
                                  trials = trials, T = T,
                                  prior_states = state_prior,
                                  policies = pol,
                                  number_of_states = ns,
                                  number_of_policies = npi)


    """
    create world
    """

    w = world.World(environment, bayes_pln, trials = trials, T = T)

    """
    simulate experiment
    """

    if not context:
        w.simulate_experiment(print_thoughts=print_thoughts)
        print("Rho", Rho)
    else:
        w.simulate_experiment(curr_trials=range(0, trials//2),print_thoughts=print_thoughts)
        Rho_new = np.zeros((nr,ns)) + const
        Rho_new[0,:] = 1 - (nr-1)*const
        Rho_new[:,g2] = [0+const, 1-(nr-1)*const]
        print("Rho_new", Rho_new)
        w.environment.Rho[:] = Rho_new
        #w.agent.perception.generative_model_rewards = Rho_new
        w.simulate_experiment(curr_trials=range(trials//2, trials))

    """
    plot and evaluate results
    """
    plt.close()
    #find successful and unsuccessful runs
    #goal = np.argmax(utility)
    successfull_g1 = np.where(environment.hidden_states[:,-1]==g1)[0]
    if context:
        successfull_g2 = np.where(environment.hidden_states[:,-1]==g2)[0]
        unsuccessfull1 = np.where(environment.hidden_states[:,-1]!=g1)[0]
        unsuccessfull2 = np.where(environment.hidden_states[:,-1]!=g2)[0]
        unsuccessfull = np.intersect1d(unsuccessfull1, unsuccessfull2)
    else:
        unsuccessfull = np.where(environment.hidden_states[:,-1]!=g1)[0]

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
    plt.savefig(path + 'grid.svg', dpi=600)
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
            d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
            if d not in [1,-1,Ly,-Ly,0]:
                print("ERROR: beaming")
            if d == 1:
                n1[environment.hidden_states[i, j],0] +=1
            if d == -1:
                n1[environment.hidden_states[i, j]-1,0] +=1
            if d == Ly:
                n1[environment.hidden_states[i, j],1] +=1
            if d == -Ly:
                n1[environment.hidden_states[i, j]-Ly,1] +=1

    n2 = np.zeros((ns, na))

    if context:
        for i in successfull_g2:

            for j in range(T-1):
                d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
                if d not in [1,-1,Ly,-Ly,0]:
                    print("ERROR: beaming")
                if d == 1:
                    n2[environment.hidden_states[i, j],0] +=1
                if d == -1:
                    n2[environment.hidden_states[i, j]-1,0] +=1
                if d == Ly:
                    n2[environment.hidden_states[i, j],1] +=1
                if d == -Ly:
                    n2[environment.hidden_states[i, j]-Ly,1] +=1

    un = np.zeros((ns, na))

    for i in unsuccessfull:

        for j in range(T-1):
            d = environment.hidden_states[i, j+1] - environment.hidden_states[i, j]
            if d not in [1,-1,Ly,-Ly,0]:
                print("ERROR: beaming")
            if d == 1:
                un[environment.hidden_states[i, j],0] +=1
            if d == -1:
                un[environment.hidden_states[i, j]-1,0] +=1
            if d == Ly:
                un[environment.hidden_states[i, j],1] +=1
            if d == -Ly:
                un[environment.hidden_states[i, j]-4,1] +=1

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
            plt.plot(x,y, 'o', color=(219/256,122/256,147/256), markersize=factor*12/(A[i,i])**2, alpha=1.)
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
    over_actions, selector, b, wd, s, A,  = par_list[1:-3]
    name = 'chosen_path'
    
    if over_actions:
        name += '_actions'
    else:
        name += '_policies' 

    if context:
        name += '_cont-1'
    else:
        name += '_cont-0'

    if prior_as_start:
        name += '_prior-1'
    else:
        name += '_prior-0'
    
    if sample_post:
        name += '_post'
    elif sample_other:
        name += '_like'
    else:
        name += '_stand'

    name += '_h'+str(h) + '_s' + str(s) + '_b' + str(b)+ '_a' + str(A) + '_' + str(run) + '.png'

    plt.savefig(path + name)


    max_RT = np.amax(w.agent.action_selection.RT[:,0])
    plt.figure()
    plt.plot(w.agent.action_selection.RT[:,0], '.')
    plt.ylim([0,1.05*max_RT])
    plt.xlim([0,trials])
    plt.savefig(path + "Gridworld_Dir_h"+str(h)+".svg")
    # plt.show()


    return w

"""
set condition dependent up parameters
"""

def run_gridworld_simulations(repetitions, print_walks=False, print_thoughts=False, my_par=None):
    # prior over outcomes: encodes utility
    utility = []

    #ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1-1e-3]
    u = 0.999
    utility = np.zeros(ns)
    utility[g1] = u
    utility[:g1] = (1-u)/(ns-1)
    utility[g1+1:] = (1-u)/(ns-1)
    

    # action selection: avergaed or max selection
    tendencies = [1,100]
    avg = True
    context = True
    if context:
        name_str = "context_"
    else:
        name_str = ""
    
    l = []                           # parameter list
    l.append([False, False, False, avg, context, utility])
    print(my_par)
    if my_par==None:
        
        my_par_list  = load_file('sim_params.txt')
        # uncertainty observation, state,
    else:
        my_par_list = my_par
    
    for indx, p in enumerate(my_par_list):
        if my_par_list[indx][0] == 3:
            my_par_list[indx][0] = True
        my_par_list[indx] = l + my_par_list[indx] 
    
    par_list = []
    
    for p in itertools.product(my_par_list, tendencies):
        par_list.append(p[0]+[p[1]])
    
    qs = [0.97]*len(par_list)


    for n,pars in enumerate(par_list):
        h = pars[-1]
        q = qs[n]
        worlds = []
        over_actions, selector, b, wd, s, A,  = pars[1:-2]
        sample_post, sample_other, prior_as_start, regime = pars[-2]

        dirname = selector + '_grid'
        if over_actions:
            dirname += '_actions'
        else:
            dirname += '_policies' 

        if context:
            dirname += '_cont-1'
        else:
            dirname += '_cont-0'

        if prior_as_start:
            dirname += '_prior-1'
        else:
            dirname += '_prior-0'
        
        if sample_post:
            dirname += '_post'
        elif sample_other:
            dirname += '_like'
        else:
            dirname += '_stand'
        
        dirname += '_h'+str(h) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
        os.mkdir(os.getcwd() + '\\agent_sims\\' + dirname)
        path = os.getcwd() + '\\agent_sims\\' + dirname + '\\'

        for i in range(repetitions):
            # dir = os.mkdir(os.getcwd() + '/agent_sims/' + 'test')
            fname = path + dirname + '_' + str(i)
            print("i", i)
            w = run_agent(pars+[q], run=i, print_thoughts = print_thoughts, print_walks = print_walks, path=path)
            
            # w = run_agent(pars+[q],var=s,run=i, sample_post=sample_post,\
            #                                   sample_other=sample_other,\
            #                                   prior_start=prior_start)
            # plot agent posterior over context

            if False:
            # if context:

                plt.figure()
                plt.plot(w.agent.posterior_context[:,0,:])
                #plt.plot(w.agent.posterior_context[:,0,g2])
                plt.title('w.agent.posterior_context[:,0,:]')
                plt.show()

                # plot reward probabilities
                plt.figure()
                rew_prob = np.einsum('tsc,tc->ts', w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0])
                rew_prob /= rew_prob.sum(axis=1)[:,None]
                plt.title('rew_prob = np.einsum("tsc,tc->ts", w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0]')
                plt.plot(rew_prob)
                plt.show()


                plt.figure()
                plt.plot(np.einsum('tsc,tc->ts', w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0]))
                plt.title('unnormalized counts for rew_prob?')
                plt.show()


            over_actions, selector, b, wd, s, A,  = pars[1:-2]
            sample_post, sample_other, prior_as_start, regime = pars[-2]
           
            save_data(fname, w)



"""
set parameters
"""

agent = 'bethe'
repetitions = 10

p = [[True, 'rdm', 3, 1, 0.0034, 1, [True, False, True, 'post_prior1']],\
     [True, 'rdm', 3, 1.5, 0.0034,  1, [True, False, True, 'post_prior1']],\
     [True, 'rdm', 3, 1, 0.0034,  0.8, [True, False, True, 'post_prior1']]]

run_gridworld_simulations(repetitions,my_par=p)

#%%
# rootdir = os.getcwd() + '\\agent_sims\\'
# print(rootdir)
# subdirs = []
# for subdir, dirs, files in os.walk(rootdir):
#     subdirs.append(subdir)
    # for file in files:
    #     print(os.path.join(subdir, file))




# sim_modes = ['\\rdm_bad_post', '\\rdm_bad_like','\\rdm_good_post','\\rdm_good_like','\\ardm_bad_post', '\\ardm_bad_like','\\ardm_good_post','\\ardm_good_like']
# sim_modes = ['\\new']
# # h = 'h1_'
# for sim_mode in sim_modes:
#     file_ttls = []
#     for subdir, dirs, files in os.walk(rootdir):
#         if subdir.__contains__(sim_mode):
#             for file in files:
#                 # if (file.__contains__(h) and not file.__contains__('.png') ):
#                 if (not file.__contains__('.png') ):
#                     file_ttls.append(file)


#     worlds = []
#     for title in file_ttls:
#         if not title == 'desktop.ini':
#             worlds.append(load_file(rootdir + sim_mode[1:] + '\\' + title))
#     # #%%
#     # import pandas as pd
#     # import seaborn as sns

#     print(worlds)
#     trials = 200
#     na = 4
#     nagents = len(worlds)
#     rt = np.zeros([nagents*trials, na])
#     agent = np.arange(nagents).repeat(trials)
#     trial = np.tile(np.arange(0,trials),nagents)
#     h = np.zeros(trials*nagents)
#     # RTs = np.arange(800).reshape([200,4])

#     names = ['agent', 'trial', 'a1','a2','a3','a4']

#     for ind, world in enumerate(worlds):
#         h[ind*trials:(ind+1)*trials] = np.min(world.agent.perception.dirichlet_pol_params).repeat(trials)
#         ac_sel = world.agent.action_selection
#         rt[ind*trials:(ind+1)*trials,:] = ac_sel.RT

#     df = pd.DataFrame(rt, columns =['a1','a2','a3','a4'])
#     df['trials'] = trial
#     df['agent'] = agent
#     df['h'] = h
# #%%
#     plt.figure()
#     sns.lineplot(data=df, x="trials",y="a1", hue="h", palette="Accent")
#     plt.savefig(rootdir + sim_mode[1:] + '\\rt_figure.png', dpi=300)
#     # plt.close()

# #%%


# my_par_list  = load_file('sim_params.txt')
# for indx, p in enumerate(my_par_list):
#     print(p)