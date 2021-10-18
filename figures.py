#%%

import seaborn as sns
import matplotlib.pyplot as plt
# from agent_simulations import make_ttl_from_params
from misc_sia import *
from misc import make_title
import numpy as np
import os as os


#%%
import matplotlib.ticker as ticker

def plot_rts_and_path(worlds, hsim, trials=5, na=4, g1=14,g2=10):
    nagents = len(worlds)
    rt = np.zeros([nagents*trials, na])
    agent = np.arange(nagents).repeat(trials)
    trial = np.tile(np.arange(0,trials),nagents)
    h = np.zeros(trials*nagents)
    prior_entropy = np.zeros(trials*nagents) 
    Q_entropy = np.zeros(trials*nagents) 

    accuracy = np.zeros(trials*nagents)
    acc = np.zeros(trials)
    goal_reached = np.zeros(trials)
    for ind, ww in enumerate(worlds):

        # print(np.min(ww.agent.perception.dirichlet_pol_params))
        h[ind*trials:(ind+1)*trials] = \
                np.min(ww.agent.perception.dirichlet_pol_params).repeat(trials)
        
        trial_index = np.arange(0,na*trials,4)
        Q = np.asarray(ww.agent.action_selection.drifts)[trial_index]
        prior = np.asarray(ww.agent.action_selection.priors)[trial_index]
        rt[ind*trials:(ind+1)*trials,:] = ww.agent.action_selection.RT
        # print(ww.agent.action_selection.RT.shape)
        Q_entropy[ind*trials:(ind+1)*trials] = entropy(Q, base=2, axis=1)
        prior_entropy[ind*trials:(ind+1)*trials] = entropy(prior, base=2, axis=1)
        goal_reached[:trials//2] = ww.environment.hidden_states[:trials//2,-1]==g1
        goal_reached[trials//2:] = ww.environment.hidden_states[trials//2:,-1]==g2
        print(ww.environment.hidden_states[:,-1])
        acc[:trials//2] = np.cumsum(goal_reached[:trials//2])
        acc[trials//2:] = np.cumsum(goal_reached[trials//2:])
        acc = acc / np.append(np.arange(2,102),np.arange(2,102))
        accuracy[ind*trials:(ind+1)*trials] = acc

    df = pd.DataFrame(rt, columns =['a1','a2','a3','a4'])
    df['trials'] = trial
    df['agent'] = agent
    df['h'] = h
    df['Q_entropy'] = Q_entropy
    df['prior_entropy'] = prior_entropy
    df['accuracy'] = accuracy

    f = plt.figure(figsize=(8,5.5))
    ax1 = f.add_subplot(2,2,1)
    sns.lineplot(data=df, x="trials",y="a1", hue="h", palette="tab10", linewidth = 1)
    ax1.set(ylabel='RT until first action')
    
    ax1 = f.add_subplot(2,2,2)
    sns.lineplot(data=df, x="trials",y="accuracy", hue="h", palette="tab10")

    axins = ax1.inset_axes([0.1, 0.1, 0.3, 0.3])
    sns.lineplot(data=df.query("trials < 115 & trials >= 98"),
             ax=axins,
             x="trials",y="accuracy", hue="h", palette="tab10", linewidth = 1)
 
    x1, x2, y1, y2 = 98, 115, -0.02, 0.07
    # sub region of the original image
    axins.set_xlim(x1, x2)  
    axins.set_ylim(y1, y2)
    
    axins.set(xlabel=None)
    axins.set(ylabel=None)

    axins.set(xticklabels=[])
    axins.set(yticklabels=[])

    axins.legend().set_visible(False)
    ax1.indicate_inset_zoom(axins, edgecolor="black")


    ax1 = f.add_subplot(2,2,3)
    sns.lineplot(data=df, x="trials",y="Q_entropy", hue="h", palette="tab10")
    ax1.set(ylabel='$H$(Q)')
    
    ax1 = f.add_subplot(2,2,4)
    sns.lineplot(data=df, x="trials",y="prior_entropy", hue="h", palette="tab10")
    ax1.set(ylabel='$H$(prior)')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.savefig(os.getcwd() + 'agent_' + hsim + '.png', dpi=300)
    plt.show()
    plt.close()



#%%



repetitions = 10

# 'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt',
#                    'npi_81_rdm_post_prior1_b_3_wd_1_s_0.001_a_4_.txt',


ttls_for_fig_81 = ['npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_.txt']
pars = []

for ttl in ttls_for_fig_81:
    pars.append(extract_params(ttl))
path = 'desired_rt_1\\'
# par_list = optimal_parameters_new
# par_list = post_sampling_params
par_list = pars     

for index, p in enumerate(par_list):
    print('currently running: ', index)
    parameters = [p]

    sim_modes = []

    for ind, p in enumerate(parameters):
        sim_modes.append(make_ttl_from_params(p))
    

    for sim_mode in sim_modes:
        worlds = []

        for hsim in sim_mode:
        #     if hsim.__contains__('_actions'):
        #         repetitions = 15
        #     else:
        #         repetitions = 10

            for r in range(repetitions):
                worlds.append(load_file(os.getcwd() + '\\agent_sims\\' + path + hsim + '_' + str(r)))


    plot_rts_and_path(worlds,hsim,trials=200)
    agent = "bethe"




extract_paths(world.environment)

# axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
# axins.imshow(Z2, extent=extent, origin="lower")
# # sub region of the original image
# x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.set_xticklabels([])
# axins.set_yticklabels([])




# ttls_for_fig = ['npi_3_rdm_standard_b_1_wd_0.1280639999999999_s_6.399999999999994e-05_.txt',
#                 'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt',
#                 'npi_3_rdm_post_prior1_b_3_wd_1_s_0.0034_a_1_.txt',
#                 'npi_81_rdm_post_prior1_b_3_wd_1_s_0.001_a_4_.txt',
#                 'npi_3_rdm_like_prior1_b_7_wd_1_s_0.0034_a_2_.txt',
#                 'npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_.txt']

# # pars = []
# # for ttl in ttls_for_fig:
# #     pars.append(extract_params(ttl))
# #%%  GOOD POST APPROXIMATION FIGURES

# optimal_parameters_new = [[3, 'rdm', 1, 2.0368092803778026, 0.005, 0.85, [False, False, True, 'standard']],
#                           [3, 'rdm', 1, 1.241687356873713, 0.005, 0.5, [True, False, True, 'post_prior1']],
#                           [3, 'rdm', 1.8, 2.059582685494318, 0.01, 0.35, [False, True, True, 'like_prior1']]]
# ttls_for_fig = []                    
# for p in optimal_parameters_new:
#     ttls_for_fig.append(make_title(p, extra_param=['a',p[5]],format='.txt'))

# ttls = np.asarray(ttls_for_fig)
# polss = np.asarray([3,8,81,2])
# regimes = ["'standard'", "'post_prior1'", "'like_prior1'"]

# fig, axes = plt.subplots(2,3, figsize=(11,5))

# npi=3
# for r, reg in enumerate(regimes):

#     ind = np.where(polss==npi)[0][0]
#     posts = np.zeros([4, polss[ind]])    # translate the posteriors
#     post = np.asarray(test_vals)[ind,:,0]    # into a numpy array

#     data = load_file(os.getcwd() + '\\parameter_data\\' + ttls[r])

#     for indx, p in enumerate(post):
#         posts[indx,:] = np.asarray(p)

#     x_positions = []
#     for i in range(4):
#         x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])
    
#     for m in range(4):
#         x_pos = x_positions[m]
#         post = posts[m]
#         axes[1,r].bar(x_pos, post, alpha=0.5, color='k', label='nolegend')
#         axes[1,r].bar(x_pos, data['empirical'][m,:], alpha=0.5, color=cols[m])        

#     rt_df = {
#         'rts': data['RT'].ravel() ,
#         'mode': tests.repeat(4000)
#         }
    
#     rt_df = pd.DataFrame(rt_df)
#     sns.histplot(ax = axes[0,r], \
#                     data=rt_df,
#                     x='rts', hue='mode', legend=False, bins=100)

# axes[0,0].set_title('Q = likelihood')
# axes[0,1].set_title('Q = posterior')
# axes[0,2].set_title('Q = prior+likelihood')

# axes[1,r].legend(labels = ['habit','goal', 'agreement','conflict'], bbox_to_anchor=(-0.1, -0.6), loc='lower center', ncol=4)
# plt.subplots_adjust(hspace=0.7)

# axes = axes[0,:].flat
# for n, ax in enumerate(axes):

#     ax.text(-0.1, 1.2, string.ascii_uppercase[n], transform=ax.transAxes, 
#             size=15)

# plt.show()
# fig.savefig('good_parametrizations.png', dpi=300)

# # #%%  GOOD RT DISTRIBUTIONS FIGURES
# # ttls = np.asarray(ttls_for_fig)
# # ttls = ttls[np.newaxis,:].reshape([3,2])
# # polss = np.asarray([3,8,81,2])
# # regimes = ["'standard'", "'post_prior1'", "'like_prior1'"]

# # fig_rt,axes_rt = plt.subplots(3,2, figsize=(8,8))
# # fig_post,axes_post = plt.subplots(3,2, figsize=(8,8), gridspec_kw={'width_ratios': [1, 3]})

# # for r, reg in enumerate(regimes):
# #     for ni, npi in enumerate([3,81]):

# #         ind = np.where(polss==npi)[0][0]
# #         posts = np.zeros([4, polss[ind]])    # translate the posteriors
# #         post = np.asarray(test_vals)[ind,:,0]    # into a numpy array
    
# #         data = load_file(os.getcwd() + '\\parameter_data\\' + ttls[r,ni])

# #         for indx, p in enumerate(post):
# #             posts[indx,:] = np.asarray(p)

# #         x_positions = []
# #         for i in range(4):
# #             x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])
        
# #         for m in range(4):
# #             x_pos = x_positions[m]
# #             post = posts[m]
# #             axes_post[r, ni].bar(x_pos, post, alpha=0.5, color='k', label='nolegend')
# #             axes_post[r, ni].bar(x_pos, data['empirical'][m,:], alpha=0.5, color=cols[m])        

# #         rt_df = {
# #             'rts': data['RT'].ravel() ,
# #             'mode': tests.repeat(1000)
# #             }
        
# #         rt_df = pd.DataFrame(rt_df)
# #         sns.histplot(ax = axes_rt[r, ni], \
# #                      data=rt_df,
# #                      x='rts', hue='mode', legend=False, bins=100)

# # axes_rt[0,0].set_title('$n$ = 3')
# # axes_rt[0,1].set_title('$n$ = 81')

# # axes_rt[r, ni].legend(labels = ['habit','goal', 'agreement','conflict'], bbox_to_anchor=(-0.1, -0.6), loc='lower center', ncol=4)
# # plt.subplots_adjust(hspace=0.6)
# # axes_rt = axes_rt.flat
# # for n, ax in enumerate(axes_rt):

# #     ax.text(-0.1, 1.2, string.ascii_uppercase[n], transform=ax.transAxes, 
# #             size=15)

# # axes_post[0,0].set_title('$n$ = 3')
# # axes_post[0,1].set_title('$n$ = 81')

# # axes_post[r, ni].legend(labels = ['habit','goal', 'agreement','conflict'], bbox_to_anchor=(0.25, -0.6), loc='lower center', ncol=4)
# # plt.subplots_adjust(hspace=0.6)
# # axes_post = axes_post.flat
# # for n, ax in enumerate(axes_post):

# #     ax.text(-0.1, 1.2, string.ascii_uppercase[n], transform=ax.transAxes, 
# #             size=15)

# # # plt.show()
# # fig_rt.savefig('example_parametrizations.png', dpi=300)
# # fig_post.savefig('example_post_approx.png', dpi=300)


        
# #%% CREATE TABLES
# # parameters = []
# # for file in ttls_for_fig:
# #     parameters.append(extract_params(file))

# # load_fits_from_ttl(ttls_for_fig):


# ttls_for_fig = ['npi_3_rdm_standard_b_1_wd_0.1280639999999999_s_6.399999999999994e-05_.txt',
#                 'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt',
#                 'npi_3_rdm_post_prior1_b_3_wd_1_s_0.0034_a_1_.txt',
#                 'npi_81_rdm_post_prior1_b_3_wd_1_s_0.001_a_4_.txt',
#                 'npi_3_rdm_like_prior1_b_7_wd_1_s_0.0034_a_2_.txt',
#                 'npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_.txt']

# pars = []
# for ttl in ttls_for_fig:
#     pars.append(extract_params(ttl))

# df,naned = load_fits_from_data(pars)

# for ind, row in df.iterrows():
#     if row['npi'] == npi:
#         print(row['regime'])


# for ind, row in df.iterrows():
#     if row['npi'] == npi:
#         print(row['stats'].to_latex())

































#%%
df.head()
df.keys()
print(df[['npi','regime','b','w','s','A']].to_latex())


#%%

optimal_parameters_new = [[3, 'rdm', 1, 2.0368092803778026, 0.005, 0.85, [False, False, True, 'standard']],
                          [3, 'rdm', 1, 1.241687356873713, 0.005, 0.5, [True, False, True, 'post_prior1']],
                          [3, 'rdm', 1.1, 5.914959918032103, 0.0064000000000000001, 0, [True, False, False, 'post_prior0']],
                          [3, 'rdm', 1.8, 2.059582685494318, 0.01, 0.35, [False, True, True, 'like_prior1']],
                          [3, 'rdm', 1, 1.7303636466497805, 0.0036, 1, [False, True, False, 'like_prior0']]]

#%%
# generate_data(optimal_parameters_new,trials=1000)

df,naned = load_fits_from_data(optimal_parameters_new)


print(df[['npi','regime','b','w','s','A']].to_latex())

# for ind, row in df.iterrows():
#     print(row['regime'])

# for ind, row in df.iterrows():

#     print(row['stats'].to_latex())

#%% LOOK AT RATIOS 

params_rt = []
params_post = optimal_parameters_new
for ttl in ttls_for_fig:
    params_rt.append(extract_params(ttl))

mode = ['rt']*6 + ['post']*5
params = params_rt + params_post

for p in params:
    p[-1] = p[-1][-1]

params = np.asarray(params)
df = pd.DataFrame(data=params, columns = ['npi','selector','b','w','s','a','regime'])
df.iloc[:,[0,2,3,4,5]] = df.iloc[:,[0,2,3,4,5]].astype('float')
df['mode'] = mode


df['drift_var_ratio'] = df['w']/df['s']