from misc import calc_dkl, extract_params_from_ttl
import pickle as pickle 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import itertools as itertools
from misc import run_action_selection, test_vals
import os 
import pandas as pd 
from scipy import stats
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


# ss = [0.01, 0.03, 0.05, 0.07, 0.1]
# wds = np.arange(0.5,2.5,0.5)
# bs = np.arange(1, 2.5, 0.5)

ss = [0.01, 0.03, 0.05, 0.07, 0.1]
wds = [0.7, 0.9, 1.3, 1.7, 1.9, 2.1, 2.3]
bs = [1, 1.3, 1.7, 1.9, 2.1, 2.3]

pols = np.array([3,81]) #,8,81]
polss =  np.array([3,8,81])
path = os.getcwd() + '\\parameter_data\\'
par_list = []

parameter_names = ['npi', 'methods', 'b', 'wd', 's', 'params_list']

for p in itertools.product(pols, methods, bs, wds, ss, params_list):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]] + [p[5]])

trials = 1000

# print(par_list)

def generate_data():

    # not currently iterating over policy sizes
    for ind, p in enumerate(par_list):
        print(ind)
        npi = p[0]
        selector = p[1]
        b = p[2]
        wd = p[3]
        s = p[4]
        sample_post, sample_other, prior_as_start, reg = p[5]

        ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 
        print(ttl)
        empirical = np.zeros([nmodes, npi])
        RT = np.zeros([nmodes, trials])

        for m, mode in enumerate(modes):
            i = np.where(polss == npi)[0][0]
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

generate_data()
def load_data():

    posts = np.zeros([len(modes), 3])    # translate the posteriors
    post = np.asarray(test_vals)[0,:,0]    # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    n = len(par_list)
    npis = np.zeros(n,dtype="int32")
    selectors = np.zeros(n).tolist()
    bs = np.zeros(n)
    wds = np.zeros(n)
    ss = np.zeros(n)
    regimes = np.zeros(n).tolist()
    post_fit = np.zeros(n)
    conf_mode = np.zeros(n)
    agr_mode = np.zeros(n)
    goal_mode = np.zeros(n)
    hab_mode = np.zeros(n)
    conf_mean = np.zeros(n)
    agr_mean = np.zeros(n)
    goal_mean = np.zeros(n)
    hab_mean = np.zeros(n)
    conf_median = np.zeros(n)
    agr_median = np.zeros(n)
    goal_median = np.zeros(n)
    hab_median = np.zeros(n)
    ttls = []

    for ind, p in enumerate(par_list):
        # print(ind)
        
        npis[ind] = p[0]
        selectors[ind] = p[1]
        bs[ind] = p[2]
        wds[ind] = p[3]
        ss[ind] = p[4]
        sample_post, sample_other, prior_as_start, regimes[ind] = p[5]


        ttl = '_'.join(['npi', str(npis[ind]), selectors[ind], regimes[ind] , 'b' ,str(bs[ind]), 'wd',\
                        str(wds[ind]), 's', str(ss[ind]), '.txt']) 
        
        with open(path + ttl, 'rb') as fp:
            data = pickle.load(fp)

        empirical = np.asarray(data['empirical'])
        RTs = np.asarray(data['RT'])
        conf_mode[ind], agr_mode[ind], goal_mode[ind], hab_mode[ind] = np.asarray(stats.mode(RTs, axis=1)[0]).ravel()
        conf_mean[ind], agr_mean[ind], goal_mean[ind], hab_mean[ind] = RTs.mean(axis=1)
        conf_median[ind], agr_median[ind], goal_median[ind], hab_median[ind] = np.median(RTs, axis=1)
        ttls.append(ttl)
        post_fit[ind] = np.abs((posts - empirical)/posts).mean(axis=1).mean()

    data = {'npi': npis,
                 'selector':selectors,
                 'b': bs,
                 'w': wds,
                 's': ss,
                 'regime': regimes,
                 'fit': post_fit,
                 'conf_mode':conf_mode,
                 'agr_mode': agr_mode,
                 'goal_mode': goal_mode,
                 'hab_mode': hab_mode,
                 'conf_mean': conf_mean,
                 'agr_mean': agr_mean,
                 'goal_mean': goal_mean,
                 'hab_mean': hab_mean,
                 'conf_median': conf_median,
                 'agr_median': agr_median,
                 'goal_median': goal_median,
                 'hab_median': hab_median,
                 'title': ttls
            }
    df = pd.DataFrame(data)
    # return best_fit, diff_best
    return df


# df = load_data()
# #  print the fit values for all the best fits
# print(df.groupby(['selector', 'regime'])['fit'].min())
# df['opt_fit_group'] = df.groupby(['npi', 'selector','regime'])['fit'].transform('min')
# df['optimal'] =(df['fit'] == df['opt_fit_group'])*1
# print(df[df['optimal'] == 1])

'''''''''''''''''''''''
'''''''''''''''''''''''


'''GET A BETTER LOOK AT RT DISTRIBUTION AND POST APPROXIMATIONS'''
# cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
# trials = 10000
# for ind, row in df[df['optimal'] == 1].iterrows():
#     npi, selector, pars, regime, s, wd, b = extract_params_from_ttl(row.title)

#     fig, ax = plt.subplots(2,1)
#     x_positions = []
#     for i in range(4):
#         x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])
    
#     for m, mode in enumerate(modes):
#         i = np.where(polss == npi)[0][0]
#         prior = test_vals[i][m][1]
#         like = test_vals[i][m][2]
#         post = test_vals[i][m][0]

#         actions, ac_sel = run_action_selection(selector, prior, like, post, trials=trials,\
#                                            prior_as_start=pars[2], sample_post=pars[0], sample_other=pars[1],\
#                                            var = s, wd =wd,b=b)

#         height = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
#         x_pos = x_positions[m]
#         lab =' '.join([mode, 'mode',  str(stats.mode(ac_sel.RT)[0][0][0]), 'median', str(np.median(ac_sel.RT)), 'mean', str(ac_sel.RT.mean())])
#         ax[0].hist(ac_sel.RT, bins=100, alpha=0.5, label=lab)
#         if m == 0:
#             ax[1].bar(x_pos, post, alpha=0.5, color='k', label = "post" )
#         else:
#                 ax[1].bar(x_pos, post, alpha=0.5, color='k')

#         ax[1].bar(x_pos, height, label=mode + ' empir', alpha=0.5, color=cols[m])


#     ttl = '_'.join(['npi', str(npi), selector, regime, 'b' ,str(b), 'wd',\
#                     str(wd), 's', str(s), '.png']) 
#     ax[0].legend()
#     ax[1].legend()
#     plt.savefig(ttl, dpi=350)

'''PLOT APPROXIMATION FROM DATA FILE AND STATS'''
# cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
# ax = plt.subplot(211)
# ax2 = plt.subplot(212)
# npi = 3
# x_positions = []

# for i in range(4):
#     x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

# with open(path + best_ttl, 'rb') as fp:
#     data = pickle.load(fp)

# empirical = np.asarray(data['empirical'])
# RT = np.asarray(data['RT'])
# df = pd.DataFrame(RT.T, index = np.arange(RT.shape[1]), columns=modes)
# print(df.describe())
# print(df.mode())
# print(df.median())
# for i in range(4):
#     x_pos = x_positions[i]
#     post = test_vals[0][i][0]
#     ax.bar(x_pos, post, label='post', alpha=0.5, color="k")
#     ax.bar(x_pos, empirical[i,:], label='empirical', alpha=0.5, color=cols[i])
#     ax2.hist(RT[i,:], bins=100, alpha=0.5)
# plt.show()



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