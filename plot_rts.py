from misc import calc_dkl
import pickle as pickle 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import itertools as itertools
from misc import run_action_selection, test_vals
import os 
'''
BRUTE FORCE PARAMETER SEARCH
'''

methods = ['rdm', 'ardm']
modes = ['conflict', 'agreement', 'goal', 'habit']
nmodes = len(modes)

params_list = [[False, False, True, 'standard'],\
               [True, False, True, 'post_prior1' ],\
               [True, False, False, 'post_prior0'],\
               [False, True, True, 'like_prior1' ],\
               [False, True, False, 'like_prior0']]

ss = [0.01, 0.03, 0.05, 0.07, 0.1]
wds = np.arange(0.5,2.5,0.5)
bs = np.arange(0.5, 2.5, 0.5)
pols = np.array([3]) #,8,81]
path = os.getcwd() + '/parameter_data/'
par_list = []

parameter_names = ['npi', 'methods', 'b', 'wd', 's', 'params_list']

for p in itertools.product(pols, methods, bs, wds, ss, params_list):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]] + [p[5]])


# print(par_list)

trials = 1000
# not currently iterating over policy sizes
for ind, p in enumerate(par_list):
    print(ind)
    npi = p[0]
    selector = p[1]
    b = p[2]
    wd = p[3]
    s = p[4]
    sample_post, sample_other, prior_as_start, reg = p[5]

    empirical = np.zeros([nmodes, npi])
    RT = np.zeros([nmodes, trials])

    for m, mode in enumerate(modes):
        i = np.where(pols == npi)[0][0]
        prior = test_vals[i][m][1]
        like = test_vals[i][m][2]
        post = test_vals[i][m][0]
        
        actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                        prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_other,\
                        var=s, wd=wd, b=b)

        empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
        RT[m,:] = ac_sel.RT.squeeze()
        

    
    ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 

    with open(path + ttl, 'wb') as fp:

        dict = {
            'RT': RT,
            'empirical': empirical,
            'parameters': parameter_names,
            'parameter_values':p
        }

        pickle.dump(dict, fp)

    # if False:
    #     with open(path + ttl, 'rb') as fp:
    #         test = pickle.load(fp)

    #     for val in test:
    #         print(val)


'''Compare posterior approximation with RT distributions'''


# vars =[0.01, 0.1]
# trials = 1000
# npi=3
# cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
# reg_titles =  ['_standard', '_post_prior1', '_post_prior0', '_like_prior1', '_like_prior0_']


# for p, pars in enumerate(params_list):

#     fig, axes = plt.subplots(2, 2, figsize=(10,5))
#     for s, var in enumerate(vars):

#         x_positions = []
#         for i in range(4):
#             x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

#         for m, mode in enumerate(modes):
#             prior = test_vals[0][m][1]
#             like = test_vals[0][m][2]
#             post = test_vals[0][m][0]

#             actions, ac_sel = run_action_selection('rdm', prior, like, post, trials=trials, prior_as_start=pars[2],\
#                                 sample_post=pars[0], sample_other=pars[1], var=var)

#             height = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
#             print(var, mode, calc_dkl(height, post))
#             x_pos = x_positions[m]
#             axes[0][s].hist(ac_sel.RT,bins=100, alpha=0.5)
#             axes[1][s].bar(x_pos, post, label='post', alpha=0.5, color="k")
#             axes[1][s].bar(x_pos, height, label='empirical', alpha=0.5, color=cols[m])

#     plt.savefig('compare_dkl_rt_rdm_npi3_b_2' + reg_titles[p]+'.png', dpi=300)



'''PLOT THE REACTION TIME DISTRUBTIONS FOR DIFFERENT VARIANCES'''
# with open('reaction_times_npi3.txt', 'rb') as fp:
#     data = pickle.load(fp)
# '''default orders from generation file RT_ddm.py
# pols = [3, 8 ,81]
# methods = ['rdm', 'ardm']
# ss = np.arange(0.01,0.11,0.01)

# params_list = [[False, False, True],\
#                [True, False, True ],\
#                [True, False, False],\
#                [False, True, True ],\
#                [False, True, False]]


# regimes = ['standard', 'post, prior as start', 'post, no start', 'like+prior, prior as start', 'like+prior, no start']
# '''

# pols = [3, 8 ,81]
# methods = ['rdm', 'ardm']
# modes = ['conflict', 'agreement', 'goal', 'habit']
# trials = 1000
# ss = np.round(np.arange(0.01,0.11,0.01),2)

# params_list = [[False, False, True],\
#                [True, False, True ],\
#                [True, False, False],\
#                [False, True, True ],\
#                [False, True, False]]


# regimes = ['standard', 'post, prior as start', 'post, no start', 'like+prior, prior as start', 'like+prior, no start']
# reg_titles =  ['_standard', '_post_prior1', '_post_prior0', '_like_prior1', '_like_prior0_']


# rdm = data[0][0]
# ardm = data[0][1]


# #  npi, selector, params, var, modes
# for p, pars in enumerate(params_list):
    
#     fig, axes = plt.subplots(2,5, figsize=(15*1.77, 15))
#     axes = axes.reshape(axes.size)

#     for s, var in enumerate(ss):
#         for m, mode in enumerate(modes):
#             mu = np.median(rdm[p][s][m]).round(2)
#             eta = np.var(rdm[p][s][m]).round(2)

#             axes[s].hist(rdm[p][s][m],bins=100, alpha=0.5, label='mu: ' + str(mu) +', s: ' + str(eta))
#             axes[s].title.set_text('var= ' + str(var))

#         axes[s].legend()
#     fig.suptitle('Reporting median and var, order conflict, agreement, goal, habit')
#     plt.savefig('rt_dist_rdm_npi3' + reg_titles[p200]+'.png', dpi=300)