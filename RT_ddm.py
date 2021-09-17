#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:10:10 2020

@author: sarah
"""
#%%
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import action_selection as asl
import seaborn as sns
import pandas as pd
from scipy.stats import entropy
plt.style.use('seaborn-whitegrid')
from pandas.plotting import table

#%%
tests = ["conflict", "agreement", "goal", "habit"]#, "uncertainty"]
num_tests = len(tests)
test_vals = [[],[],[]]


#////////// setup 2 policies
npi = 2
flat = [1./npi]*npi

# conflict
l = [0.8,0.2]
conflict_prior = np.ones(2) - l + 0.1
conflict_prior /= conflict_prior.sum()

prior = np.array(conflict_prior)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# agreement
l = np.array([0.8,0.2])
agree_prior = l + 0.3
agree_prior /= agree_prior.sum()

prior = np.array(agree_prior)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# goal
l = [0.8,0.2]
prior = np.array(flat)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# habit
prior = np.array([0.8,0.2])
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])



# ///////// setup 8 policies

npi = 8
gp = 2
high_prob_1 = 0.4
high_prob_2 = 0.3
flat = np.ones(npi)/npi
h1 = np.array([high_prob_1]*gp + [(1 - high_prob_1*gp)/(npi-gp)]*(npi-gp))
h2_conf = np.array([(1 - high_prob_2*gp)/(npi-gp)]*gp + [high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp*2))
h2_agree = np.array([high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp))

# conflict
prior = h1.copy()
like = h2_conf.copy()
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])

# agreement
prior = h1.copy()
like = h2_agree.copy()
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])

#goal
prior = flat
like = h1.copy()
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])

# habit
prior = h1.copy()
like = flat
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])


#//// setup 81 policies
num_tests = len(tests)
gp = 6
n = 81
val = 0.148
l = [val]*gp+[(1-6*val)/(n-gp)]*(n-gp)
v = 0.00571
p = [(1-(v*(n-gp)))/6]*gp+[v]*(n-gp)
conflict = [v]*gp+[(1-(v*(n-gp)))/6]*gp+[v]*(n-2*gp)
npi = n
flat = [1./npi]*npi

# conflict
prior = np.array(conflict)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])

# agreement
prior = np.array(p)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])

# goal
prior = np.array(flat)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])

# habit
prior = np.array(l)
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])



# for y in range(3):
#     print(y)
#     fig, ax = plt.subplots(2,2, figsize = (8,6))
#     ax = ax.reshape(ax.size)
#     for i in range(4):
#         ax[i].plot(test_vals[y][i][0], label='posterior')
#         ax[i].plot(test_vals[y][i][1], label='prior')
#         ax[i].plot(test_vals[y][i][2], label='likelihood')
#         ax[i].set_title(tests[i])

    # plt.close()

#%% FUNCTIONS 

def create_titles(mode, selector):

    if (selector.type == 'rdm'):
        fig_title = mode + ', ' + 'wd: ' + str(selector.wd) + \
                    ', ' + 's: ' + str(selector.s) + \
                    ', ' + 'b: ' + str(selector.b)
    
        file_title = mode + '_' + 'npi-' + str(npi) + \
                    '_' + 'wd-' + str(selector.wd) + \
                    '_' + 's-' + str(selector.s) + \
                    '_' + 'b-' + str(selector.b) + '.png'
        
    
    elif (selector.type == 'ardm'):
        fig_title = mode + ', ' + 'wd: ' + str(selector.wd) + \
                           ', ' + 'ws: ' + str(selector.ws) + \
                           ', ' + 's: ' + str(selector.s) + \
                           ', ' + 'b: ' + str(selector.b)

        file_title = mode + '_' + 'npi-' + str(npi) + \
                    '_' + 'wd-' + str(selector.wd) + \
                    '_' + 'ws-' + str(selector.ws) + \
                    '_' + 's-' + str(selector.s) + \
                    '_' + 'b-' + str(selector.b) + '.png'
        
    return fig_title,file_title


# test function                                          #sort of like urgency factor? 
# def run_action_selection(post, prior, like, trials = 100, crit_factor = 0.5, calc_dkl = False):

#     ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor, calc_dkl=calc_dkl)
#     for t in range(trials):
#         ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior)

#     if calc_dkl:
#         return ac_sel.RT.squeeze(), ac_sel.DKL_post.squeeze(), ac_sel.DKL_prior.squeeze()
#     else:
#         return ac_sel.RT.squeeze()

# set up number of trials



def run_action_selection(post, prior, like, selector, mode, crit_factor= 0.5, trials = 100, T = 2, plotting=False):
    
    if (selector == 'ddm'): 
        ac_sel = asl.DDM_RandomWalker(trials,T, s = 0.01)
    elif (selector == 'rdm'):
        ac_sel = asl.RacingDiffusionSelector(trials,T, s=0.03)
    elif (selector == 'ardm'):
        ac_sel = asl.AdvantageRacingDiffusionSelector(trials, T)
    elif (selector == 'dirichlet'):
        ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor, calc_dkl=False)
    else:
        raise ValueError('selector not given properly')
    
    print(trials)
    actions = []

    for t in range(trials):
       
        if (selector == 'rdm' or selector == 'ardm'):

            actions.append(ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior))
            # if (t%10 == 0 and (selector == 'rdm' or selector == 'ardm')):
            #     save_trajectory(mode, ac_sel, selector,t)

        elif(selector == "ddm"):

            actions.append(ac_sel.select_desired_action(t, 0, prior, like, post,True))
        else:

            ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior, plot=plotting)
            
    return ac_sel.RT.squeeze(), ac_sel, actions

def save_trajectory(mode,selector,selector_type,t):
    plt.close()
    path = '/home/terraregina/Documents/university/MSc/habit/project/sarah_code/BalancingControl-reaction_times/' + mode + '/'
    plt.plot(selector.trajectory)
    plt.savefig(path + selector_type +'_'+'npi-'+str(npi)+'_'+mode+str(t)+'.png',dpi=100)



# #%% Simulation with RDM 
# trials = 100
# cols = ['r','g','b','k']
# modes = len(test_vals[0])
# titles = ['conflict', 'agreement', 'goal','habit']
# selector = 'rdm'
# npi = [2,8,81]

# for pol_regime in range(1):                       #for 2,8,81 policies
#     RTs = []
#     curr_test_vals = np.asarray(test_vals[pol_regime])
#     npi = curr_test_vals.shape[2]

#     for mode in range(modes):                                  #for conf,agreement...etc
#         print(selector,'policy regime:',pol_regime, titles[mode])
#         post = curr_test_vals[mode,0,:]
#         prior = curr_test_vals[mode,1,:]
#         like = curr_test_vals[mode,2,:]
#         rt, sampler, actions = run_action_selection(post, prior, like, selector, titles[mode], trials=trials, plotting=True)
#         RTs.append(rt)


#         plt.close()
#         # save example walks
#         for i in range(len(sampler.walks)):
#             plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i][:,0], color="k")
#             plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i][:,1], color = "r")
#         ttl = "walks_" + titles[mode] +"_npi-"+ str(npi)+"_s-"+str(sampler.s) + "_w-"+str(sampler.wd)+".png"
#         plt.savefig(ttl)
        
        
#         plt.close()
#         fig,axs = plt.subplots(1,3, figsize=(7,4))
        
#         #plot rt histogram
#         axs[0].plot(post, label="post")
#         axs[0].plot(prior, label = "prior")
#         axs[0].plot(like, label = "like")
#         axs[0].legend()

#         axs[1].hist(rt, bins=400, label=titles[mode], color=cols[mode])
#         axs[1].legend()

#         # plot actions and how they approximate posterior over actions

#         height = np.asarray([np.asarray(actions).sum(), trials - np.asarray(actions).sum()])/trials
#         bars = ('0', '1')

#         # Choose the width of each bar and their positions
#         width = [1,1]
#         x_pos = [0,1]
        
#         # Make the plot
#         axs[2].bar(x_pos, height, width=width, alpha=0.8, label="empirical")
#         axs[2].bar(x_pos, post,width=width, alpha=0.5, label="post")
#         # axs[2].set_xticks(x_pos, bars)
#         axs[2].legend()
#         ttl = selector+"_npi-2_" + titles[mode] + '.png'
#         plt.savefig(ttl)

#     df = pd.DataFrame(np.asarray(RTs).T, columns=titles)

#     # plot and save
#     plt.close()
#     fig_ttl, file_ttl = create_titles(selector, sampler)
#     plt.figure(figsize=(16, 12.5))
#     ax = []
#     miniax = []

#     miniax.append(plt.subplot(4, 4, 1))
#     miniax.append(plt.subplot(4, 4, 2))
#     miniax.append(plt.subplot(4, 4, 5))
#     miniax.append(plt.subplot(4, 4, 6))

#     ax.append(miniax)
#     ax.append(plt.subplot(2, 2, 2))
#     ax.append(plt.subplot(2, 1, 2))


#     for i in range(4):
#         ax[0][i].plot(curr_test_vals[i][0], 'o--', label='posterior')
#         ax[0][i].plot(curr_test_vals[i][1], 'o--', label='prior')
#         ax[0][i].plot(curr_test_vals[i][2], 'o--', label='likelihood')
#         ax[0][i].set_title(titles[i])

#     df.boxplot(ax=ax[1], showmeans=True)

#     for i in range(modes):
#         ax[2].hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
#     ax[2].legend(loc="lower center",ncol=len(df.columns),bbox_to_anchor=(0.5, -0.1))

#     plt.savefig(file_ttl, dpi=100)

#%% Simulation with DDM 2 policies

# vals = np.asarray(test_vals[0])
# modes = vals.shape[0]
# RTs = []
# titles = ['conflict', 'agreement', 'goal', 'habit']
# cols = ['r','g','b','k']
# trials = 1000
# plt.figure()
# selector = "ddm"
# plt.close()
# for mode in range(4):
#     # simulate reaction times
#     post = vals[mode,0,:]
#     prior = vals[mode,1,:]
#     like = vals[mode,2,:]
#     rt, sampler, actions = run_action_selection(post, prior, like, selector,titles[mode], trials=trials)

#     RTs.append(rt)

#     fig,axs = plt.subplots(1,3, figsize=(7,4))
    
#     #plot rt histogram
#     axs[0].plot(post, label="post")
#     axs[0].plot(prior, label = "prior")
#     axs[0].plot(like, label = "like")
#     axs[0].legend()

#     axs[1].hist(rt, bins=100, label=titles[mode], color=cols[mode])
#     axs[1].legend()

#     # # plot walks
#     # for i in range(len(sampler.walks)):
#     #     plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i])
#     # plt.show()

#     # plot actions and how they approximate posterior over actions

#     height = np.asarray([trials - np.asarray(actions).sum(), np.asarray(actions).sum()])/trials
#     bars = ('0', '1')

#     # Choose the width of each bar and their positions
#     width = [1,1]
#     x_pos = [0,1]
    
#     # Make the plot
#     axs[2].bar(x_pos, height, width=width, alpha=0.8, label="empirical")
#     axs[2].bar(x_pos, post,width=width, alpha=0.5, label="post")
#     # axs[2].set_xticks(x_pos, bars)
#     axs[2].legend()
#     ttl = 'ddm_'+"npi-2_" + titles[mode] + '.png'
#     plt.savefig(ttl)





#%% Simulation with RDM 
trials = 1000
cols = ['r','g','b','k']
modes = len(test_vals[0])
titles = ['conflict', 'agreement', 'goal','habit']
selectors = ['rdm']
# selectors = ['rdm','ardm','dirichlet']
npi = [2,8,81]
for sel in range(len(selectors)):                                  #for each action selection type
    selector = selectors[sel]
    for pol_regime in range(1):                       #for 2,8,81 policies
        RTs = []
        curr_test_vals = np.asarray(test_vals[pol_regime])
        npi = curr_test_vals.shape[2]

        for mode in range(modes):                                  #for conf,agreement...etc
            print(selectors[sel],'policy regime:',pol_regime, titles[mode])
            post = curr_test_vals[mode,0,:]
            prior = curr_test_vals[mode,1,:]
            like = curr_test_vals[mode,2,:]
            rt, sampler, actions = run_action_selection(post, prior, like, selector, titles[mode], trials=trials, plotting=True)
            RTs.append(rt)
            plt.close()
            # save example walks
            for i in range(len(sampler.walks)):
                plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i])
            ttl = "walks_" + titles[mode] +"_npi-"+ str(npi)+"_s-"+str(sampler.s) + "_w-"+str(sampler.wd)+".png"
            plt.savefig(ttl)
            plt.close()

            fig,axs = plt.subplots(1,3, figsize=(7,4))
            
            #plot rt histogram
            axs[0].plot(post, label="post")
            axs[0].plot(prior, label = "prior")
            axs[0].plot(like, label = "like")
            axs[0].legend()

            axs[1].hist(rt, bins=400, label=titles[mode], color=cols[mode])
            axs[1].legend()

            # plot actions and how they approximate posterior over actions

            height = np.asarray([trials - np.asarray(actions).sum(), np.asarray(actions).sum()])/trials
            bars = ('0', '1')

            # Choose the width of each bar and their positions
            width = [1,1]
            x_pos = [0,1]
            
            # Make the plot
            axs[2].bar(x_pos, height, width=width, alpha=0.8, label="empirical")
            axs[2].bar(x_pos, post,width=width, alpha=0.5, label="post")
            # axs[2].set_xticks(x_pos, bars)
            axs[2].legend()
            ttl = selector+"_npi-2_" + titles[mode] + '.png'
            plt.savefig(ttl)

        df = pd.DataFrame(np.asarray(RTs).T, columns=titles)

        # plot and save
        plt.close()
        fig_ttl, file_ttl = create_titles(selector, sampler)
        plt.figure(figsize=(16, 12.5))
        ax = []
        miniax = []

        miniax.append(plt.subplot(4, 4, 1))
        miniax.append(plt.subplot(4, 4, 2))
        miniax.append(plt.subplot(4, 4, 5))
        miniax.append(plt.subplot(4, 4, 6))

        ax.append(miniax)
        ax.append(plt.subplot(2, 2, 2))
        ax.append(plt.subplot(2, 1, 2))


        for i in range(4):
            ax[0][i].plot(curr_test_vals[i][0], 'o--', label='posterior')
            ax[0][i].plot(curr_test_vals[i][1], 'o--', label='prior')
            ax[0][i].plot(curr_test_vals[i][2], 'o--', label='likelihood')
            ax[0][i].set_title(titles[i])

        df.boxplot(ax=ax[1], showmeans=True)

        for i in range(modes):
            ax[2].hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
        ax[2].legend(loc="lower center",ncol=len(df.columns),bbox_to_anchor=(0.5, -0.1))

        plt.savefig(file_ttl, dpi=100)


# # ######### PLOT RT DISTRIBUTIONS DIRICHLET
# test_vals = np.asarray(test_vals)
# modes = test_vals.shape[0]
# RTs = []
# titles = ['conflict', 'agreement', 'goal', 'habit']

# plt.figure()

# for mode in range(modes):

#     post = test_vals[mode,0,:]
#     prior = test_vals[mode,1,:]
#     like = test_vals[mode,2,:]
#     rt = run_action_selection(post, prior, like, trials,  crit_factor=0.5,calc_dkl=False)
#     RTs.append(rt)

# for i in range(modes):
#     plt.hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
# plt.legend()


# df = pd.DataFrame(np.asarray(RTs).T, columns=titles)
# print(df.describe())

# %%

#%% simulate ARDM
# test_vals = np.asarray(test_vals)
# modes = test_vals.shape[0]
# RTs = []
# titles = ['conflict', 'agreement', 'goal','habit']
# selector = 'rdm'

# for mode in range(modes):
#     print(titles[mode])
#     post = test_vals[mode,0,:]
#     prior = test_vals[mode,1,:]
#     like = test_vals[mode,2,:]
#     rt, sampler = run_action_selection_ddm(post, prior, like, selector, titles[mode], trials, plotting=True)
#     RTs.append(rt)

# df = pd.DataFrame(np.asarray(RTs).T, columns=titles)

# # plot and save
# fig_ttl, file_ttl = create_titles(selector, sampler)
# plt.figure(figsize=(16, 12.5))
# ax = []
# miniax = []

# miniax.append(plt.subplot(4, 4, 1))
# miniax.append(plt.subplot(4, 4, 2))
# miniax.append(plt.subplot(4, 4, 5))
# miniax.append(plt.subplot(4, 4, 6))

# ax.append(miniax)
# ax.append(plt.subplot(2, 2, 2))
# ax.append(plt.subplot(2, 1, 2))


# for i in range(4):
#     ax[0][i].plot(curr_test_vals[i][0], 'o--', label='posterior')
#     ax[0][i].plot(curr_test_vals[i][1], 'o--', label='prior')
#     ax[0][i].plot(curr_test_vals[i][2], 'o--', label='likelihood')
#     ax[0][i].set_title(titles[i])

# df.boxplot(ax=ax[1], showmeans=True)

# for i in range(modes):
#     ax[2].hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
# ax[2].legend(loc="lower center",ncol=len(df.columns),bbox_to_anchor=(0.5, -0.1))

# plt.savefig(file_ttl, dpi=100)



