


#%%
import numpy as np
import seaborn as sns
import os as os 
from misc import load_file,params_list
from scipy.stats import entropy
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import pandas as pd
"""
Goal: plot the entropies of the relevant decision making distributions
prior and Q 
plot average over available runs 

look at rdm bad like where there is no peak at 100 trial but a decrease in the beginning 
and a continuously decreasing habit
and rdm bad post where there is no mega pronounced dip at 100 for
"""

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

def make_ttl_from_params(p):
    context = True
    over_actions, selector, b, wd, s, A,  = p[:-1]
    sample_post, sample_other, prior_as_start, regime = p[-1]
    # s = round(s,5)

    if over_actions == 3:
        over_actions = True
    elif over_actions == 81:
        over_actions = False
        
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
    
    low = dirname + '_h'+str(1) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
    high = dirname + '_h'+str(1000) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
    return [low, high]

    # high = dirname + '_h'+str(1000) + '_s' + str(s)+ '_wd' + str(wd) + '_b' + str(b) + '_a' + str(A)
    # return [high]

# load relevant data
rootdir = os.getcwd() + '\\agent_sims\\'
# sim_modes = ['\\rdm_grid_actions_cont-1_prior-1_post_h1_s0.0004_wd1_b3_a1','\\rdm_grid_actions_cont-1_prior-1_post_h1000_s0.0004_wd1_b3_a1' ]
# p = [[True, 'rdm', 3, 1, 0.0034, 1, [True, False, True, 'post_prior1']]]
# params = load_file('standard_params.txt')


# s = np.asarray([0.02, 0.04, 0.06])
# s = s**2
# params = [[81, 'rdm', 1, 1, s[0], 1, params_list[1]],
#      [81, 'rdm', 1, 1, s[0], 1, params_list[3]],
#      [81, 'rdm', 1, 1, s[1], 1, params_list[1]],
#      [81, 'rdm', 1, 1, s[1], 1, params_list[3]],
#      [81, 'rdm', 1, 1, s[2], 1, params_list[1]],
#      [81, 'rdm', 1, 1, s[2], 1, params_list[3]]]

# params = [[3, 'rdm', 1, 1, s[2], 1, params_list[3]]]

# pars_for_fig = ['npi_3_rdm_standard_b_1_wd_0.1280639999999999_s_6.399999999999994e-05_a_1_.txt', 'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt', 'npi_3_rdm_post_prior1_b_3_wd_1_s_0.0034_a_1_.txt', 'npi_81_rdm_post_prior1_b_3_wd_1_s_0.001_a_4_.txt', 'npi_3_rdm_like_prior1_b_7_wd_1_s_0.0034_a_2_.txt', 'npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_.txt']	
# pars_for_fig = ['npi_3_rdm_standard_b_1_wd_1.5_s_0.0016_a_1_.txt']
# pars_for_fig = ['npi_3_rdm_standard_b_1_wd_0.5_s_0.005_a_1_.txt']
# good_fit = ['npi_3_rdm_like_prior1_b_2.5_wd_0.906_s_0.006_.txt']


# pars = []
# for par in pars_for_fig:
#     pars.append(extract_params(par))

# sim_modes = []

# for ind, p in enumerate(pars):
#     sim_modes.append(make_ttl_from_params(p))

#%%

for sim_mode in sim_modes:
    worlds = []

    for hsim in sim_mode:

        file_ttls = []
        for subdir, dirs, files in os.walk(rootdir):
            if subdir.__contains__(hsim):
                for file in files:
                    # if (file.__contains__(h) and not file.__contains__('.png') ):
                    if (not file.__contains__('.png') and not file.__contains__('.svg')):
                        file_ttls.append(file)


        for title in file_ttls:
            if not title == 'desktop.ini':
                worlds.append(load_file(rootdir + hsim + '\\' + title))


# extract prior over actions and relevant q

    trials = 200
    na = 4
    nagents = len(worlds)
    rt = np.zeros([nagents*trials, na])
    agent = np.arange(nagents).repeat(trials)
    trial = np.tile(np.arange(0,trials),nagents)
    h = np.zeros(trials*nagents)
    prior_entropy = np.zeros(trials*nagents) 
    Q_entropy = np.zeros(trials*nagents) 

    for ind, world in enumerate(worlds):

        print(np.min(world.agent.perception.dirichlet_pol_params))
        h[ind*trials:(ind+1)*trials] = \
                np.min(world.agent.perception.dirichlet_pol_params).repeat(trials)
        
        trial_index = np.arange(0,800,4)
        Q = np.asarray(world.agent.action_selection.drifts)[trial_index]
        prior = np.asarray(world.agent.action_selection.priors)[trial_index]
        rt[ind*trials:(ind+1)*trials,:] = world.agent.action_selection.RT
        # print(world.agent.action_selection.RT.shape)
        Q_entropy[ind*trials:(ind+1)*trials] = entropy(Q, base=2, axis=1)
        prior_entropy[ind*trials:(ind+1)*trials] = entropy(prior, base=2, axis=1)

    df = pd.DataFrame(rt, columns =['a1','a2','a3','a4'])
    df['trials'] = trial
    df['agent'] = agent
    df['h'] = h
    df['Q_entropy'] = Q_entropy
    df['prior_entropy'] = prior_entropy

    f = plt.figure(figsize=(12,3))
    ax1 = f.add_subplot(1,3,1)
    sns.lineplot(data=df, x="trials",y="a1", hue="h", palette="Dark2")
    ax1 = f.add_subplot(1,3,2)
    sns.lineplot(data=df, x="trials",y="Q_entropy", hue="h", palette="Dark2")
    ax1 = f.add_subplot(1,3,3)
    sns.lineplot(data=df, x="trials",y="prior_entropy", hue="h", palette="Dark2")

    plt.savefig(rootdir + hsim + '\\rt_figure_line.png', dpi=300)
    # plt.show()
    plt.close()

    break

#%%

df