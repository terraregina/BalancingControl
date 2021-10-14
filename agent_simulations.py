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
import pickle as pickle
import itertools as itertools
import numpy as np
from scipy.stats import entropy
sns.set_style("whitegrid")
from misc_sia import load_fits, plot_rts_and_entropy
#%%
save = True
data_folder = os.path.join('C:\\Users\\admin\\Desktop\\project\\BalancingControl','data')

const = 0#1e-10

trials = 200   #number of trials
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


def extract_params(ttl):
    names = ['standard', 'post_prior1', 'post_prior0', 'like_prior1', 'like_prior0']

    params_dict = {
        'standard_b': [False, False, True],
        'post_prior1': [True, False, True], 
        'post_prior0': [True, False, False],
        'like_prior1': [False, True, True], 
        'like_prior0': [False, True, False]
    }
    pars = ttl.split('_')
    a_present = False
    for indx, par in enumerate(pars):
        if par == 'b':
            if not len(pars[indx+1]) == 1: 
                b = float(pars[indx+1])
            else:
                b = int(pars[indx+1])
        if par == 's':
            s = float(pars[indx+1])
        if par == 'wd':
            if not len(pars[indx+1]) == 1: 
                wd = float(pars[indx+1])
            else:
                wd = int(pars[indx+1])
        if par == 'a':
            a_present = True
            if not len(pars[indx+1]) == 1: 
                a = float(pars[indx+1])
            else:
                a = int(pars[indx+1])
            # a = float(pars[indx+1])
    # print(pars)            
    npi = int(pars[1])
    selector = pars[2]
    regime = '_'.join(pars[3:5])
    pars = params_dict[regime]
    
    if regime == 'standard_b':
        regime = 'standard'
    if a_present:
        return [npi, selector, b, wd,s, a, pars + [regime]]
    else:
        return [npi, selector, b, wd,s, 1, pars + [regime]]

#%%
def make_ttl_from_params(p):
    context = True
    over_actions, selector, b, wd, s, A,  = p[:-1]
    sample_post, sample_other, prior_as_start, regime = p[-1]

    # if over_actions == 3:
    #     over_actions = True
    # else: 
    #     over_actions = False
        
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
    
    # low = dirname + '_h'+str(1) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
    high = dirname + '_h'+str(1000) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
    # return [low, high]

    high = dirname + '_h'+str(1000) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
    return [high]

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

    if selector == 'avg':
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
    # fig = plt.figure(figsize=[factor*5,factor*4])

    # ax = fig.gca()

    # annot = np.zeros((Lx,Ly))
    # for i in range(Lx):
    #     for j in range(Ly):
    #         annot[i,j] = i*Ly+j

    # u = sns.heatmap(start_goal, ax = ax, **grid_plot_kwargs, annot=annot, annot_kws={"fontsize": 40})
    # ax.invert_yaxis()
    # plt.savefig(path + 'grid.svg', dpi=600)
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

    name += '_h'+str(h) + '_s' + str(s) + '_b' + str(b)+ '_wd' + str(wd) + '_a' + str(A) + '_' + str(run) + '.png'

    plt.savefig(path + name)


    max_RT = np.amax(w.agent.action_selection.RT[:,0])
    # plt.figure()
    # plt.plot(w.agent.action_selection.RT[:,0], '.')
    # plt.ylim([0,1.05*max_RT])
    # plt.xlim([0,trials])
    # plt.savefig(path + "Gridworld_Dir_h"+str(h)+".svg")
    # plt.show()


    return w

"""
set condition dependent up parameters
"""

def run_gridworld_simulations(repetitions, print_walks=False, print_thoughts=False, my_par=None, file_name=None):
    # prior over outcomes: encodes utility
    utility = []

    #ut = [0.5, 0.6, 0.7, 0.8, 0.9, 1-1e-3]
    u = 0.999
    utility = np.zeros(ns)
    utility[g1] = u
    utility[:g1] = (1-u)/(ns-1)
    utility[g1+1:] = (1-u)/(ns-1)
    

    # action selection: avergaed or max selection
    tendencies = [1000]
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
        
        my_par_list  = load_file(file_name)
        # uncertainty observation, state,
    else:
        my_par_list = my_par
    
    print(my_par_list)
    for indx, p in enumerate(my_par_list):
        if my_par_list[indx][0] == 3:
            my_par_list[indx][0] = True
        else:
            my_par_list[indx][0] = False

        my_par_list[indx] = l + my_par_list[indx]
    
    
    par_list = []
    
    for p in itertools.product(my_par_list, tendencies):
        par_list.append(p[0]+[p[1]])
    
    qs = [0.97]*len(par_list)


    for n,pars in enumerate(par_list):
        h = pars[-1]
        q = qs[n]
        worlds = []
        print(pars)
        over_actions, selector, b, wd, s, A,  = pars[1:-2]
        sample_post, sample_other, prior_as_start, regime = pars[-2]
        print(pars)
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
        path = os.getcwd() + '\\agent_sims\\'

        for i in range(repetitions):
            fname = path + dirname + '_' + str(i)
            print(dirname + '_' + str(i))
            print("i", i)
            w = run_agent(pars+[q], run=i, print_thoughts = print_thoughts, print_walks = print_walks, path=path)
            

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
                plt.title('rew_prob avg= np.einsum("tsc,tc->ts", w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0]')
                plt.plot(rew_prob)
                plt.show()


                plt.figure()
                plt.plot(np.einsum('tsc,tc->ts', w.agent.posterior_dirichlet_rew[:,0,1,:,:],w.agent.posterior_context[:,0]))
                plt.title('unnormalized counts for rew_prob?')
                plt.show()


            over_actions, selector, b, wd, s, A,  = pars[1:-2]
            sample_post, sample_other, prior_as_start, regime = pars[-2]
           
            save_data(fname, w)
        
    return my_par_list


"""
set parameters
"""

agent = 'bethe'
repetitions = 3

# p = [[True, 'rdm', 3, 1, 0.0034, 1, [True, False, True, 'post_prior1']]]
# p = [[True, 'rdm', 1, 1, 0.0004, 1, [False, True,  True, 'like_prior1']]]

# p = [[True, 'rdm', 1, 1, 0.0004, 1, [True, False, True, 'post_prior1']]]
# p = [[True, 'ardm', 1, 1, 0.0004, 1, [True, False, True, 'post_prior1']]]
# p = [[True, 'ardm', 1, 1, 0.0004, 1, [False, True, True, 'like_prior1']]]
# p = [[True, 'rdm', 1, 1, 0.001, 1, [True, False, True, 'post_prior1']]]
# p = [[True, 'rdm', 1, 1, 0.0004, 1, params_list[0]]]
# p = [[True, 'rdm', 1, 1, 0.001, 1, params_list[0]]]


# pars = load_file('standard_params.txt')

# params = []
# for i in [1, 2, 3, 4, 5]: 
# #     params.append(pars[i])
# p =[[81, 'rdm', 1.0, 0.1280639999999999, 6.399999999999994e-05, 3.5, [False, False, True, 'standard']]]
# p =[[81, 'rdm', 1.0, 0.1280639999999999, 6.399999999999994e-05, 3.5, [False, False, True, 'standard']]]
# s = np.asarray([0.02, 0.04, 0.06])
# s = s**2
# params = [[3, 'rdm', 1, 1, s[0], 1, params_list[1]],
#      [3, 'rdm', 1, 1, s[0], 1, params_list[3]],
#      [3, 'rdm', 1, 1, s[1], 1, params_list[1]],
#      [3, 'rdm', 1, 1, s[1], 1, params_list[3]],
#      [3, 'rdm', 1, 1, s[2], 1, params_list[1]],
#      [3, 'rdm', 1, 1, s[2], 1, params_list[3]],
#      [81, 'rdm', 1, 1, s[0], 1, params_list[1]],
#      [81, 'rdm', 1, 1, s[0], 1, params_list[3]],
#      [81, 'rdm', 1, 1, s[1], 1, params_list[1]],
#      [81, 'rdm', 1, 1, s[1], 1, params_list[3]],
#      [81, 'rdm', 1, 1, s[2], 1, params_list[1]],
#      [81, 'rdm', 1, 1, s[2], 1, params_list[3]]]

# pars_for_fig = ['npi_3_rdm_standard_b_1_wd_0.1280639999999999_s_6.399999999999994e-05_a_1_.txt', 'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt', 'npi_3_rdm_post_prior1_b_3_wd_1_s_0.0034_a_1_.txt', 'npi_81_rdm_post_prior1_b_3_wd_1_s_0.001_a_4_.txt', 'npi_3_rdm_like_prior1_b_7_wd_1_s_0.0034_a_2_.txt', 'npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_.txt']	
# pars_for_fig = ['npi_3_rdm_standard_b_1_wd_0.5_s_0.005_a_1_.txt']

# good_fit = ['npi_3_rdm_like_prior0_b_2.5_wd_1.809_s_0.009_.txt', 'npi_3_rdm_like_prior1_b_2.5_wd_0.906_s_0.006_.txt', 'npi_3_rdm_post_prior0_b_2.5_wd_1.197_s_0.007_.txt', 'npi_3_rdm_post_prior1_b_1_wd_20.05_s_0.05_.txt', 'npi_3_rdm_standard_b_1_wd_22.55_s_0.05_.txt']
# pars = []

# for par in good_fit:
    # pars.append(extract_params(par))

# par_list = run_gridworld_simulations(repetitions,file_name ='standard_params.txt')

#%%
# df = load_fits()
#%%%
# df2 = df[0]
# df2.query("selector == 'rdm' ")
# stopped = 128


# from a paparameter list
selectors = ['rdm']
npi = [3,81]
ss = np.asarray([0.01, 0.03, 0.05, 0.07, 0.1])**2
# ss = np.asarray([0.07])**2

ws = [1, 1.5, 2, 2.3]
bs = [1, 1.5, 2.0, 2.5]

par_list = []
params_list2 = params_list[1:]

for p in itertools.product(selectors, npi, params_list2, bs, ws, ss, [1]):
        par_list.append([p[1]]+ [p[0]] + [p[3]]+ [p[4]]+ [p[5]] + [p[6]] + [p[2]])


for index, p in enumerate(par_list):
    print('currently running: ', index)
    parameters = [p]
    pars = run_gridworld_simulations(1, my_par=parameters)

    sim_modes = []

    for ind, p in enumerate(parameters):
        sim_modes.append(make_ttl_from_params(p[1:]))

    #%%

    for sim_mode in sim_modes:
        worlds = []

        for hsim in sim_mode:
            worlds.append(load_file(os.getcwd() + '\\agent_sims\\' + hsim + '_0'))

        path = os.getcwd() + '\\agent_sims\\'

        plot_rts_and_entropy(worlds,hsim)
        os.remove(os.getcwd() + '\\agent_sims\\' + hsim + '_0')
        agent = "bethe"






# # from a dataframe 
# for ind, row in df2.iterrows():
#     parameters = []
#     parameters.append(extract_params(row['file_ttl']))
#     pars = run_gridworld_simulations(1, my_par=parameters)

#     sim_modes = []

#     for ind, p in enumerate(parameters):
#         sim_modes.append(make_ttl_from_params(p[1:]))

#     #%%

#     for sim_mode in sim_modes:
#         worlds = []
#         for hsim in sim_mode:
#             worlds.append(load_file(os.getcwd() + '\\agent_sims\\' + hsim + '_0'))


#         # for hsim in sim_mode:

#         #     file_ttls = []
#         #     for subdir, dirs, files in os.walk(rootdir):
#         #         if subdir.__contains__(hsim):
#         #             for file in files:
#         #                 # if (file.__contains__(h) and not file.__contains__('.png') ):
#         #                 if (not file.__contains__('.png') and not file.__contains__('.svg')):
#         #                     file_ttls.append(file)


#         #     for title in file_ttls:
#         #         if not title == 'desktop.ini':
#         #             worlds.append(load_file(rootdir + hsim + '\\' + title))

#         path = os.getcwd() + '\\agent_sims\\'



#%%

# rootdir = os.getcwd() + '\\agent_sims\\'
# print(rootdir)
# subdirs = []
# for subdir, dirs, files in os.walk(rootdir):
#     subdirs.append(subdir)
#     print(dirs)
# #     for file in files:
# #         print(os.path.join(subdir, file))



# # sim_modes = []
# # with open('sim_params.txt', 'rb') as fp:
# #     par_list = pickle.load(fp)

# # for pars in par_list[:1]:

# #     sim_modes += make_ttl_from_params(pars)

# # print(sim_modes)

# # sim_modes = ['rdm_grid_actions_cont-1_prior-1_stand_h1000_s0.0004_wd1_b3_a1','rdm_grid_actions_cont-1_prior-1_post_h1000_s0.0004_wd1_b3_a1','rdm_grid_actions_cont-1_prior-0_post_h1000_s0.0004_wd1_b3_a1']
# # print(sim_modes)
# # # h = 'h1_'
# p  =[[81, 'rdm', 1.0, 0.1280639999999999, 6.399999999999994e-05, 3.5, [False, False, True, 'standard']]]
# sim_modes = make_ttl_from_params(p[0])

# worlds = []
# for sim_mode in sim_modes:
#     file_ttls = []
#     for subdir, dirs, files in os.walk(rootdir):
#         if subdir.__contains__(sim_mode):
#             for file in files:
#                 # if (file.__contains__(h) and not file.__contains__('.png') ):
#                 if (not file.__contains__('.png') and not file.__contains__('.svg')):
#                     file_ttls.append(file)

#     for title in file_ttls:
#         print(title)
#         if not title == 'desktop.ini':
#             worlds.append(load_file(rootdir + sim_mode + '\\' + title))




# # print(worlds)
# # trials = 200 
# # na = 4
# # nagents = len(worlds)
# # rt = np.zeros([nagents*trials, na])
# # agent = np.arange(nagents).repeat(trials)
# # trial = np.tile(np.arange(0,trials),nagents)
# # h = np.zeros(trials*nagents)
# # # RTs = np.arange(800).reshape([200,4])

# # names = ['agent', 'trial', 'a1','a2','a3','a4']

# # for ind, world in enumerate(worlds):
# #     print(np.min(world.agent.perception.dirichlet_pol_params))
# #     h[ind*trials:(ind+1)*trials] = np.min(world.agent.perception.dirichlet_pol_params).repeat(trials)
# #     ac_sel = world.agent.action_selection
# #     rt[ind*trials:(ind+1)*trials,:] = ac_sel.RT

# # df = pd.DataFrame(rt, columns =['a1','a2','a3','a4'])
# # df['trials'] = trial
# # df['agent'] = agent
# # df['h'] = h

# # plt.figure()
# # sns.lineplot(data=df, x="trials",y="a1", hue="h", palette="Accent")
# # plt.savefig(rootdir + sim_mode + '\\rt_figure.png', dpi=300)
# # plt.show()
# # plt.close()




