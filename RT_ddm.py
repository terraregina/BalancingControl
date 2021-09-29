# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Thu Dec 10 15:10:10 2020

# @author: sarah
# """
# #%%
# from numpy.lib.npyio import save
# import pandas as pd
# import numpy as np
# import matplotlib.pylab as plt
# import action_selection as asl
# import seaborn as sns
# import pandas as pd
# from scipy.stats import entropy
# plt.style.use('seaborn-whitegrid')
# from pandas.plotting import table
# import os

# #%%


# tests = ["conflict", "agreement", "goal", "habit"]#, "uncertainty"]
# num_tests = len(tests)
# test_vals = [[],[],[]]


# #////////// setup 2 policies
# npi = 2
# flat = [1./npi]*npi

# # conflict
# l = [0.8,0.2]
# conflict_prior = np.ones(2) - l + 0.1
# conflict_prior /= conflict_prior.sum()

# prior = np.array(conflict_prior)
# like = np.array(l)
# post = prior*like
# post /= post.sum()
# test_vals[0].append([post,prior,like])

# # agreement
# l = np.array([0.8,0.2])
# agree_prior = l + 0.3
# agree_prior /= agree_prior.sum()

# prior = np.array(agree_prior)
# like = np.array(l)
# post = prior*like
# post /= post.sum()
# test_vals[0].append([post,prior,like])

# # goal
# l = [0.8,0.2]
# prior = np.array(flat)
# like = np.array(l)
# post = prior*like
# post /= post.sum()
# test_vals[0].append([post,prior,like])

# # habit
# prior = np.array([0.8,0.2])
# like = np.array(flat)
# post = prior*like
# post /= post.sum()
# test_vals[0].append([post,prior,like])



# # ///////// setup 8 policies

# npi = 8
# gp = 2
# high_prob_1 = 0.4
# high_prob_2 = 0.3
# flat = np.ones(npi)/npi
# h1 = np.array([high_prob_1]*gp + [(1 - high_prob_1*gp)/(npi-gp)]*(npi-gp))
# h2_conf = np.array([(1 - high_prob_2*gp)/(npi-gp)]*gp + [high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp*2))
# h2_agree = np.array([high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp))

# # conflict
# prior = h1.copy()
# like = h2_conf.copy()
# post = prior*like
# post /= post.sum()
# test_vals[1].append([post,prior,like])

# # agreement
# prior = h1.copy()
# like = h2_agree.copy()
# post = prior*like
# post /= post.sum()
# test_vals[1].append([post,prior,like])

# #goal
# prior = flat
# like = h1.copy()
# post = prior*like
# post /= post.sum()
# test_vals[1].append([post,prior,like])

# # habit
# prior = h1.copy()
# like = flat
# post = prior*like
# post /= post.sum()
# test_vals[1].append([post,prior,like])


# #//// setup 81 policies
# num_tests = len(tests)
# gp = 6
# n = 81
# val = 0.148
# l = [val]*gp+[(1-6*val)/(n-gp)]*(n-gp)
# v = 0.00571
# p = [(1-(v*(n-gp)))/6]*gp+[v]*(n-gp)
# conflict = [v]*gp+[(1-(v*(n-gp)))/6]*gp+[v]*(n-2*gp)
# npi = n
# flat = [1./npi]*npi

# # conflict
# prior = np.array(conflict)
# like = np.array(l)
# post = prior*like
# post /= post.sum()
# test_vals[2].append([post,prior,like])

# # agreement
# prior = np.array(p)
# like = np.array(l)
# post = prior*like
# post /= post.sum()
# test_vals[2].append([post,prior,like])

# # goal
# prior = np.array(flat)
# like = np.array(l)
# post = prior*like
# post /= post.sum()
# test_vals[2].append([post,prior,like])

# # habit
# prior = np.array(l)
# like = np.array(flat)
# post = prior*like
# post /= post.sum()
# test_vals[2].append([post,prior,like])



# #%% FUNCTIONS 

# def create_titles(mode, selector):

#     if (selector.type == 'rdm'):
#         fig_title = mode + ', ' + 'wd: ' + str(selector.wd) + \
#                     ', ' + 's: ' + str(selector.s) + \
#                     ', ' + 'b: ' + str(selector.b)
    
#         file_title = mode + '_' + 'npi-' + str(npi) + \
#                     '_' + 'wd-' + str(selector.wd) + \
#                     '_' + 's-' + str(selector.s) + \
#                     '_' + 'b-' + str(selector.b) + '.png'
        
    
#     elif (selector.type == 'ardm'):
#         fig_title = mode + ', ' + 'wd: ' + str(selector.wd) + \
#                            ', ' + 'ws: ' + str(selector.ws) + \
#                            ', ' + 's: ' + str(selector.s) + \
#                            ', ' + 'b: ' + str(selector.b)

#         file_title = mode + '_' + 'npi-' + str(npi) + \
#                     '_' + 'wd-' + str(selector.wd) + \
#                     '_' + 'ws-' + str(selector.ws) + \
#                     '_' + 's-' + str(selector.s) + \
#                     '_' + 'b-' + str(selector.b) + '.png'
        
#     return fig_title,file_title


# # test function                                          #sort of like urgency factor? 
# # def run_action_selection(post, prior, like, trials = 100, crit_factor = 0.5, calc_dkl = False):

# #     ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor, calc_dkl=calc_dkl)
# #     for t in range(trials):
# #         ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior)

# #     if calc_dkl:
# #         return ac_sel.RT.squeeze(), ac_sel.DKL_post.squeeze(), ac_sel.DKL_prior.squeeze()
# #     else:
# #         return ac_sel.RT.squeeze()

# # set up number of trials



# def run_action_selection(post, prior, like, selector, mode,\
#                          crit_factor= 0.5, trials = 100, T = 2,\
#                          plotting=False, s = 0.1, sample_posterior = False,\
#                          sample_other = False, prior_as_starting_point=True):
    
#     if (selector == 'ddm'): 
#         ac_sel = asl.DDM_RandomWalker(trials,T, s=s)
#     elif (selector == 'rdm'):
#         # def __init__ (self, trials, T, number_of_actions=2, wd = 1, s = 0.05, b = 1, A = 1, v0 = 0):
#         ac_sel = asl.RacingDiffusionSelector(trials,T,s=s)
#         ac_sel.sample_posterior = sample_posterior
#         ac_sel.prior_as_starting_point = prior_as_starting_point
#         ac_sel.sample_other = sample_other
#     elif (selector == 'ardm'):
#         ac_sel = asl.AdvantageRacingDiffusionSelector(trials, T)
#     elif (selector == 'dirichlet'):
#         ac_sel = asl.DirichletSelector(trials, 2, npi, factor=crit_factor, calc_dkl=False)
#     else:
#         raise ValueError('selector not given properly')
    
#     # print(trials)
#     actions = []

#     for t in range(trials):

#         if (selector == 'rdm' or selector == 'ardm'):

#             actions.append(ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior))
#             # if (t%10 == 0 and (selector == 'rdm' or selector == 'ardm')):
#             #     save_trajectory(mode, ac_sel, selector,t)

#         elif(selector == "ddm"):

#             actions.append(ac_sel.select_desired_action(t, 0, prior, like, post,True))
#         else:

#             ac_sel.select_desired_action(t, 0, post, list(range(npi)), like, prior, plot=plotting)
            
#     return ac_sel.RT.squeeze(), ac_sel, actions

# def save_trajectory(mode,selector,selector_type,t):
#     plt.close()
#     path = '/home/terraregina/Documents/university/MSc/habit/project/sarah_code/BalancingControl-reaction_times/' + mode + '/'
#     plt.plot(selector.trajectory)
#     plt.savefig(path + selector_type +'_'+'npi-'+str(npi)+'_'+mode+str(t)+'.png',dpi=100)





# # #%% RDM, all policies, different variances, SAMPLING FROM POSTERIOR, USE AS STARTING POINT

# # path = os.getcwd() + '/rdm/sample_posterior/'
# # plt.close()
# # pols = [2,8,81]
# # selector = "rdm"
# # titles = ['conflict', 'agreement', 'goal', 'habit']
# # cols = ['r','g','b','k']
# # alphas = [0.2,0.4, 0.7, 0.9]

# # for prior in range(2):

# #     prior_as_starting_point = bool(prior)

# #     for p in range(0,3):

# #         npi = pols[p]
# #         vals = np.asarray(test_vals[p])
# #         modes = vals.shape[0]
# #         RTs_overall = []

# #         x_positions = []
# #         for i in range(4):
# #             x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

# #         trials = 1000

# #         ss = [0.1,0.05, 0.03, 0.02]

# #         for mode in range(modes):

# #             prior = vals[mode,1,:]
# #             like = vals[mode,2,:]
# #             post = vals[mode,0,:]

# #             fig,axs = plt.subplots(1,3, figsize=(10,4))
# #             fig.suptitle('RDM, sample from posterior, use prior: ' + str(prior_as_starting_point) +', ' + str(npi) + ' policies, ' + titles[mode])
# #             #plot rt histogram
# #             axs[0].plot(prior, label = "prior")
# #             axs[0].plot(like, label = "like")
# #             axs[0].plot(post, label="post")
# #             axs[0].legend()

# #             RTs = []
# #             for s in range(len(ss)):

# #                 # simulate reaction times
# #                 rt, sampler, actions = run_action_selection(post, prior, like, selector,titles[mode], trials=trials, s=ss[s],\
# #                                                             sample_posterior=True, prior_as_starting_point=prior_as_starting_point)
# #                 # print(titles[mode])
# #                 # print(actions)
# #                 RTs.append(rt)

# #                 lab = 's = ' + str(ss[s])
# #                 axs[1].hist(rt, bins=100, label= lab,alpha=alphas[s], range=[0,400])
# #                 axs[1].legend()

# #                 # # plot walks
# #                 # for i in range(len(sampler.walks)):
# #                 #     plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i])
# #                 # plt.show()

# #                 # plot actions and how they approximate posterior over actions
# #                 #  had to introduce fake occurance of each action to make sure they are in there
# #                 height = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)

# #                 bars = ('0', '1')

# #                 # Choose the width of each bar and their positions
# #                 width = [1]*npi
# #                 x_pos = x_positions[s]

# #                 # Make the plot
# #                 if s == 0:
# #                     axs[2].bar(x_pos, height, width=width, alpha=0.8, label="empirical", color = 'C0')
# #                     axs[2].bar(x_pos, post,width=width, alpha=0.5, label="post", color = 'C1')
# #                     # axs[2].set_xticks(x_pos, bars)
# #                     axs[2].legend()
# #                 else:
# #                     axs[2].bar(x_pos, height, width=width, alpha=0.8, color = 'C0')
# #                     axs[2].bar(x_pos, post,width=width, alpha=0.5, color = 'C1')
            
# #                 ttl = 'rdm_'+"npi-" + str(npi) + '_' + titles[mode] + '_prior_' + str(prior_as_starting_point) + '_sample_posterior.png'
# #                 ttl = path + ttl
# #                 plt.savefig(ttl,dpi=300)
            
# #             RTs_overall.append(RTs)
# #             RTs = []



# #         # plot according to mode

# #         RT_overall = np.array(RTs_overall)
# #         alphas = [0.2,0.4, 0.7, 0.9]
# #         alphas = [0.9, 0.7, 0.4, 0.4]
# #         fig, ax = plt.subplots(1,4, figsize=(10,4))

# #         m_c = 0
# #         for rt_mode in RT_overall:

# #             s_c = 0
# #             for rt_s in rt_mode:
# #                 ax[s_c].hist(rt_s, bins=100, alpha=alphas[m_c], range=(0,600), label=titles[m_c])
# #                 s_c+=1
# #             s_c = 0
# #             # if(m_c == 2):
# #             #     break
# #             m_c += 1

# #         for i in range(len(titles)):
# #             ax[i].title.set_text('variance = ' + str(ss[i]))
# #         plt.legend()
# #         ttl = path + 'combined_rdm_npi-' + str(npi) + '_prior-' + str(prior_as_starting_point) + '_sample_posterior.png'
# #         plt.savefig(ttl, dpi=300)

# #%% RDM, all policies, different variances, SAMPLING FROM Likelihood + prior

# path = os.getcwd() + '/rdm/sample_other/'
# plt.close()
# pols = [2,8,81]
# selector = "rdm"
# titles = ['conflict', 'agreement', 'goal', 'habit']
# cols = ['r','g','b','k']
# alphas = [0.2,0.4, 0.7, 0.9]

# for prior in range(2):

#     prior_as_starting_point = bool(prior)

#     for p in range(2,3):

#         npi = pols[p]
#         vals = np.asarray(test_vals[p])
#         modes = vals.shape[0]
#         RTs_overall = []

#         x_positions = []
#         for i in range(4):
#             x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

#         trials = 1000

#         ss = [0.1,0.05, 0.03, 0.02]

#         for mode in range(modes):

#             prior = vals[mode,1,:]
#             like = vals[mode,2,:]
#             post = vals[mode,0,:]

#             fig,axs = plt.subplots(1,3, figsize=(10,4))
#             fig.suptitle('RDM, sample from sum of likelihood and prior, use prior: ' + str(prior_as_starting_point) +', ' + str(npi) + ' policies, ' + titles[mode])
#             #plot rt histogram
#             axs[0].plot(prior, label = "prior")
#             axs[0].plot(like, label = "like")
#             axs[0].plot(post, label="post")
#             axs[0].legend()

#             RTs = []
#             for s in range(len(ss)):

#                 # simulate reaction times
#                 rt, sampler, actions = run_action_selection(post, prior, like, selector,titles[mode], trials=trials, s=ss[s],\
#                                                             sample_other=True, prior_as_starting_point=prior_as_starting_point)
#                 # print(titles[mode])
#                 # print(actions)
#                 RTs.append(rt)

#                 lab = 's = ' + str(ss[s])
#                 axs[1].hist(rt, bins=100, label= lab,alpha=alphas[s], range=[0,400])
#                 axs[1].legend()

#                 # # plot walks
#                 # for i in range(len(sampler.walks)):
#                 #     plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i])
#                 # plt.show()

#                 # plot actions and how they approximate posterior over actions
#                 #  had to introduce fake occurance of each action to make sure they are in there
#                 height = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)

#                 bars = ('0', '1')

#                 # Choose the width of each bar and their positions
#                 width = [1]*npi
#                 x_pos = x_positions[s]

#                 # Make the plot
#                 if s == 0:
#                     axs[2].bar(x_pos, height, width=width, alpha=0.8, label="empirical", color = 'C0')
#                     axs[2].bar(x_pos, post,width=width, alpha=0.5, label="post", color = 'C1')
#                     # axs[2].set_xticks(x_pos, bars)
#                     axs[2].legend()
#                 else:
#                     axs[2].bar(x_pos, height, width=width, alpha=0.8, color = 'C0')
#                     axs[2].bar(x_pos, post,width=width, alpha=0.5, color = 'C1')
            
#                 ttl = 'rdm_'+"npi-" + str(npi) + '_' + titles[mode] + '_prior_' + str(prior_as_starting_point) + '_sample_other.png'
#                 ttl = path + ttl
            
#             plt.savefig(ttl,dpi=300)
            
#             RTs_overall.append(RTs)
#             RTs = []



#         # plot according to mode

#         RT_overall = np.array(RTs_overall)
#         alphas = [0.2,0.4, 0.7, 0.9]
#         alphas = [0.9, 0.7, 0.4, 0.4]
#         fig, ax = plt.subplots(1,4, figsize=(10,4))
#         table = np.zeros([4,4,2])
#         m_c = 0

#         for ind, rt_mode in enumerate(RT_overall):
#             s_c = 0                  # iterating over variances
#             for rt_s in rt_mode:
#                 table[ind,s_c,0] = np.mean(rt_s)
#                 table[ind,s_c,1] = np.var(rt_s)
#                 ax[s_c].hist(rt_s, bins=100, alpha=alphas[m_c], range=(0,600), label=titles[m_c])
#                 s_c+=1
#             s_c = 0
#             # if(m_c == 2):
#             #     break
#             m_c += 1
#         lims = [200, 200, 200, 300]
#         for i in range(len(titles)):
#             ax[i].title.set_text('variance = ' + str(ss[i]))
#             ax[i].set_xlim(0,lims[i])
#         plt.legend()

#         print(table[:,:,0])
#         print(table[:,:,1])
#         ttl = path + 'combined_rdm_npi-' + str(npi) + '_prior-' + str(prior_as_starting_point) + '_sample_other.png'
#         plt.savefig(ttl, dpi=300)



# #%% Simulation with RDM 
# trials = 1000
# cols = ['r','g','b','k']
# modes = len(test_vals[0])
# titles = ['conflict', 'agreement', 'goal','habit']
# selectors = ['rdm']
# # selectors = ['rdm','ardm','dirichlet']
# npi = [2,8,81]
# for sel in range(len(selectors)):                                  #for each action selection type
#     selector = selectors[sel]
#     for pol_regime in range(1):                       #for 2,8,81 policies
#         RTs = []
#         curr_test_vals = np.asarray(test_vals[pol_regime])
#         npi = curr_test_vals.shape[2]

#         for mode in range(modes):                                  #for conf,agreement...etc
#             print(selectors[sel],'policy regime:',pol_regime, titles[mode])
#             post = curr_test_vals[mode,0,:]
#             prior = curr_test_vals[mode,1,:]
#             like = curr_test_vals[mode,2,:]
#             rt, sampler, actions = run_action_selection(post, prior, like, selector, titles[mode], trials=trials, plotting=True)
#             RTs.append(rt)
#             plt.close()
#             # save example walks
#             for i in range(len(sampler.walks)):
#                 plt.plot(range(0,len(sampler.walks[i])), sampler.walks[i])
#             ttl = "walks_" + titles[mode] +"_npi-"+ str(npi)+"_s-"+str(sampler.s) + "_w-"+str(sampler.wd)+".png"
#             plt.savefig(ttl)
#             plt.close()

#             fig,axs = plt.subplots(1,3, figsize=(7,4))
            
#             #plot rt histogram
#             axs[0].plot(post, label="post")
#             axs[0].plot(prior, label = "prior")
#             axs[0].plot(like, label = "like")
#             axs[0].legend()

#             axs[1].hist(rt, bins=400, label=titles[mode], color=cols[mode])
#             axs[1].legend()

#             # plot actions and how they approximate posterior over actions

#             height = np.asarray([trials - np.asarray(actions).sum(), np.asarray(actions).sum()])/trials
#             bars = ('0', '1')

#             # Choose the width of each bar and their positions
#             width = [1,1]
#             x_pos = [0,1]
            
#             # Make the plot
#             axs[2].bar(x_pos, height, width=width, alpha=0.8, label="empirical")
#             axs[2].bar(x_pos, post,width=width, alpha=0.5, label="post")
#             # axs[2].set_xticks(x_pos, bars)
#             axs[2].legend()
#             ttl = selector+"_npi-2_" + titles[mode] + '.png'
#             plt.savefig(ttl)

#         df = pd.DataFrame(np.asarray(RTs).T, columns=titles)

#         # plot and save
#         plt.close()
#         fig_ttl, file_ttl = create_titles(selector, sampler)
#         plt.figure(figsize=(16, 12.5))
#         ax = []
#         miniax = []

#         miniax.append(plt.subplot(4, 4, 1))
#         miniax.append(plt.subplot(4, 4, 2))
#         miniax.append(plt.subplot(4, 4, 5))
#         miniax.append(plt.subplot(4, 4, 6))

#         ax.append(miniax)
#         ax.append(plt.subplot(2, 2, 2))
#         ax.append(plt.subplot(2, 1, 2))


#         for i in range(4):
#             ax[0][i].plot(curr_test_vals[i][0], 'o--', label='posterior')
#             ax[0][i].plot(curr_test_vals[i][1], 'o--', label='prior')
#             ax[0][i].plot(curr_test_vals[i][2], 'o--', label='likelihood')
#             ax[0][i].set_title(titles[i])

#         df.boxplot(ax=ax[1], showmeans=True)

#         for i in range(modes):
#             ax[2].hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
#         ax[2].legend(loc="lower center",ncol=len(df.columns),bbox_to_anchor=(0.5, -0.1))

#         plt.savefig(file_ttl, dpi=100)


# # # ######### PLOT RT DISTRIBUTIONS DIRICHLET
# # test_vals = np.asarray(test_vals)
# # modes = test_vals.shape[0]
# # RTs = []
# # titles = ['conflict', 'agreement', 'goal', 'habit']

# # plt.figure()

# # for mode in range(modes):

# #     post = test_vals[mode,0,:]
# #     prior = test_vals[mode,1,:]
# #     like = test_vals[mode,2,:]
# #     rt = run_action_selection(post, prior, like, trials,  crit_factor=0.5,calc_dkl=False)
# #     RTs.append(rt)

# # for i in range(modes):
# #     plt.hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
# # plt.legend()


# # df = pd.DataFrame(np.asarray(RTs).T, columns=titles)
# # print(df.describe())

# # %%

# #%% simulate ARDM
# # test_vals = np.asarray(test_vals)
# # modes = test_vals.shape[0]
# # RTs = []
# # titles = ['conflict', 'agreement', 'goal','habit']
# # selector = 'rdm'

# # for mode in range(modes):
# #     print(titles[mode])
# #     post = test_vals[mode,0,:]
# #     prior = test_vals[mode,1,:]
# #     like = test_vals[mode,2,:]
# #     rt, sampler = run_action_selection_ddm(post, prior, like, selector, titles[mode], trials, plotting=True)
# #     RTs.append(rt)

# # df = pd.DataFrame(np.asarray(RTs).T, columns=titles)

# # # plot and save
# # fig_ttl, file_ttl = create_titles(selector, sampler)
# # plt.figure(figsize=(16, 12.5))
# # ax = []
# # miniax = []

# # miniax.append(plt.subplot(4, 4, 1))
# # miniax.append(plt.subplot(4, 4, 2))
# # miniax.append(plt.subplot(4, 4, 5))
# # miniax.append(plt.subplot(4, 4, 6))

# # ax.append(miniax)
# # ax.append(plt.subplot(2, 2, 2))
# # ax.append(plt.subplot(2, 1, 2))


# # for i in range(4):
# #     ax[0][i].plot(curr_test_vals[i][0], 'o--', label='posterior')
# #     ax[0][i].plot(curr_test_vals[i][1], 'o--', label='prior')
# #     ax[0][i].plot(curr_test_vals[i][2], 'o--', label='likelihood')
# #     ax[0][i].set_title(titles[i])

# # df.boxplot(ax=ax[1], showmeans=True)

# # for i in range(modes):
# #     ax[2].hist(RTs[i], alpha=(1-0.2*i), label=titles[i])
# # ax[2].legend(loc="lower center",ncol=len(df.columns),bbox_to_anchor=(0.5, -0.1))

# # plt.savefig(file_ttl, dpi=100)



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
import pickle


# DEFINE PRIORS AND LIKELIHOODDS

tests = ["conflict", "agreement", "goal", "habit"]
num_tests = len(tests)
test_vals = [[],[],[]]

# ///////// setup 2 policies

npi = 3
gp = 1
high_prob_1 = 0.7
high_prob_2 = 0.7
flat = np.ones(npi)/npi
h1 = np.array([high_prob_1]*gp + [(1 - high_prob_1*gp)/(npi-gp)]*(npi-gp))
h2_conf = np.array([(1 - high_prob_2*gp)/(npi-gp)]*gp + [high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp*2))
h2_agree = np.array([high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp))

# conflict
prior = h1.copy()
like = h2_conf.copy()
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# agreement
prior = h1.copy()
like = h2_agree.copy()
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

#goal
prior = flat
like = h1.copy()
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# habit
prior = h1.copy()
like = flat
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])


############# setup 8 policies

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


############# setup 81 policies
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


plot = False
if plot:

    for y in range(3):
        # print(y)
        fig, ax = plt.subplots(2,2, figsize = (8,6))
        ax = ax.reshape(ax.size)
        for i in range(4):
            ax[i].plot(test_vals[y][i][0], label='posterior')
            ax[i].plot(test_vals[y][i][1], label='prior')
            ax[i].plot(test_vals[y][i][2], label='likelihood')
            ax[i].set_title(tests[i])

        # plt.close()


def run_action_selection(selector, prior, like, post, trials=10, T=2, prior_as_start=True, sample_post=False, sample_other=False, var=0.01):
    
    na = prior.shape[0]
    controls = np.arange(0, na, 1)
    if selector == 'rdm':
        # not really over_actions, simply avoids passing controls
        ac_sel = asl.RacingDiffusionSelector(trials, T, number_of_actions=na, s=var, over_actions=False)
    elif selector == 'ardm':
        # not really over_actions, simply avoids passing controls
        ac_sel = asl.AdvantageRacingDiffusionSelector(trials, T, number_of_actions=na, s=var, over_actions=False)
    else: 
        raise ValueError('Wrong or no action selection method passed')
    
    ac_sel.prior_as_starting_point = prior_as_start
    ac_sel.sample_other = sample_other
    ac_sel.sample_posterior = sample_post
    # print('prior as start, sample_other, sample_post')
    # print(ac_sel.prior_as_starting_point, ac_sel.sample_other, ac_sel.sample_posterior)
    actions = []
    
    for trial in range(trials):
        actions.append(ac_sel.select_desired_action(trial, 0, post, controls, like, prior))   #trial, t, post, control, like, prior


    return actions, ac_sel

    #%% RDM, 81 policies, different variances, SAMPLING FROM POSTERIOR

plt.close()
pols = [3, 8 ,81]
methods = ['rdm', 'ardm']
modes = ['conflict', 'agreement', 'goal', 'habit']
cols = ['r','g','b','k']
alphas = [0.2,0.4, 0.7, 0.9]
trials = 1000
ss = np.arange(0.01,0.11,0.01)

params_list = [[False, False, True],\
               [True, False, True ],\
               [True, False, False],\
               [False, True, True ],\
               [False, True, False]]


regimes = ['standard', 'post, prior as start', 'post, no start', 'like+prior, prior as start', 'like+prior, no start']

RTs_overall = []

for pi,npi in enumerate(pols):
    rt_sel = []
    print(npi)
    for sel, selector in enumerate(methods):
        print(selector)

        fig, axes = plt.subplots(len(params_list), len(ss), figsize=(20,17))
  
        rt_par = []
        
        dists = np.asarray(test_vals[pi])
        for p, pars in enumerate(params_list):    
            print('params = ', pars)
            rt_vars = []
            
            for s,var in enumerate(ss): 
              
                # if s == 0:
                #     axes[p,0].set_ylabel(regimes[p])
            
                rt_mode = []


                for m, mode in enumerate(modes):
                    post = dists[m,0]
                    prior = dists[m,1]
                    like = dists[m,2]
                    actions, sl = run_action_selection(selector, prior, like, post, trials=trials,\
                                                        prior_as_start=pars[2], sample_post=pars[0], sample_other=pars[1], var=var)

                    rt_mode.append(sl.RT)

                    # axes[p,s].hist(sl.RT, bins=100, color=cols[m], alpha=alphas[m], label=mode)
            
                # if p == len(params_list)-1:
                #     axes[p,s].set_xlabel('var=' + str(var))    

                #     if s == len(ss) - 1:
                #         axes[p,s].legend()

                rt_vars.append(rt_mode)
                rt_mode = []
            
            
            rt_par.append(rt_vars)
            rt_vars = []
    
        rt_sel.append(rt_par)
        rt_par = []

        # ttl = 'combined_npi'+str(npi) +'_' + selector + '.png'
        # plt.savefig(ttl,dpi=300)
 
    RTs_overall.append(rt_sel)
    rt_sel = []

    ttl = 'reaction_times_npi' + str(npi) + '.txt'

    with open(ttl, 'wb') as fp:
        pickle.dump(RTs_overall, fp)

    print('saved data for ', npi)




# plot according to mode

# RT_overall = np.array(RTs_overall)
# alphas = [0.2,0.4, 0.7, 0.9]
# alphas = [0.9, 0.7, 0.4, 0.4]
# fig, ax = plt.subplots(1,4, figsize=(10,4))

# m_c = 0
# for rt_mode in RT_overall:

#     s_c = 0
#     for rt_s in rt_mode:
#         ax[s_c].hist(rt_s, bins=100, alpha=alphas[m_c], range=(0,600), label=titles[m_c])
#         s_c+=1
#     s_c = 0
#     # if(m_c == 2):
#     #     break
#     m_c += 1

# for i in range(len(titles)):
#     ax[i].title.set_text('variance = ' + str(ss[i]))
# plt.legend()
# plt.savefig('rdm_npi-81_combined.png', dpi=300)
