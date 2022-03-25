import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import pandas as pd
import cufflinks as cf
import json as js
import itertools

from venv import create
from matplotlib.style import context
import os
import action_selection as asl
import seaborn as sns
from itertools import product
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json

import perception as prc
import agent as agt
from environment import PlanetWorld
from agent import BayesianPlanner
from world import World
from planet_sequences import generate_trials_df


def run_agent(par_list, trials, T, ns=6, na=2, nr=3, nc=2, npl=2):

    # learn_pol          = initial concentration parameters for POLICY PRIOR
    # context_trans_prob = probability of staing in a given context, int
    # avg                = average vs maximum selection, True for avg
    # Rho                = environment reward generation probability as a function of time, dims: trials x nr x ns
    # utility            = GOAL PRIOR, preference p(o)
    # B                  = state transition matrix depedenent on action: ns x ns x actions
    # npl                = number of unique planets accounted for in the reward contingency representation  
    # C_beta           = phi or estimate of p(reward|planet,context) 
    
    learn_pol, context_trans_prob, cue_ambiguity, avg, Rho, utility, B, planets, starts, colors, rc, learn_rew = par_list


    """
    create matrices
    """

    #generating probability of observations in each state
    A = np.eye(ns)


    # agent's initial estimate of reward generation probability
    C_beta = rc.copy()
 
    C_agent = np.zeros(C_beta.shape)        # nr x npl; default order is r=(1, 0, 1) and s=(0,1,2) 

    for c in range(nc):
        C_agent[:,:,c] = np.array([(C_beta[:,i,c])/(C_beta[:,i,c]).sum() for i in range(npl)]).T


    """
    initialize context transition matrix
    """

    p = context_trans_prob
    q = (1-p)/(nc-1)

    transition_matrix_context = np.zeros([nc,nc]) + q
    transition_matrix_context = transition_matrix_context - np.eye(nc)*q + np.eye(nc)*p 



    """ 
    create environment class
    """
    
    environment = PlanetWorld(A,
                              B,
                              Rho,
                              planets,
                              starts,
                              colors,
                              trials,
                              T,
                              ns,
                              npl,
                              nr,
                              na)


    """ 
    create policies and setup concentration parameters
    The pseudo counts alpha^t_ln which parameterize the prior over actions for
    """

    pols = np.array(list(itertools.product(list(range(na)), repeat=T-1)))
    npi = pols.shape[0]


    C_alphas = np.zeros((npi, nc)) + learn_pol
    prior_pi = C_alphas / C_alphas.sum(axis=0)

    """
    set state prior
    """

    state_prior = np.ones((ns))
    state_prior = state_prior/ state_prior.sum() 


    """
    set action selection method
    """

    if avg:

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

    """
    set context prior
    """

    prior_context = np.zeros((nc)) + 0.1/(nc-1)
    prior_context[0] = 0.9

    """
    NEED TO ALSO DEFINE A DISTRIBUTION OF p(O|C) for the color
    """


    no =  np.unique(colors).size                    # number of observations (background color)
    C = np.zeros([no, nc])
    p = cue_ambiguity                    
    dp = 0.001
    p2 = 1 - p - dp
    C[0,:] = [p,dp/2,p2,dp/2]
    C[1,:] = [dp/2, p, dp/2, p2]


    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, 
                                           B, 
                                           C_agent, 
                                           transition_matrix_context, 
                                           state_prior, 
                                           utility, 
                                           prior_pi, 
                                           C_alphas,
                                           C_beta,
                                           generative_model_context = C,
                                           T=T)


    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc,
                                    ac_sel,
                                    pols,
                                    prior_states = state_prior,
                                    prior_policies = prior_pi,
                                    trials = trials,
                                    prior_context = prior_context,
                                    learn_habit=True,
                                    learn_rew = learn_rew,
                                    number_of_planets = npl
                                    )


    """
    create world
    """

    w = World(environment, bayes_pln, trials = trials, T = T)
    bayes_pln.world = w

    """
    simulate experiment
    """

    w.simulate_experiment(range(trials))
    w.h = learn_pol
    w.q = context_trans_prob
    w.p = cue_ambiguity

    return w


def run_single_sim(lst,
                                ns,
                                na,
                                npl,
                                nc,
                                nr,
                                T,
                                state_transition_matrix,
                                planet_reward_probs,
                                planet_reward_probs_switched,
                                repetitions):

    folder = os.getcwd()
    switch_cues, contingency_degradation, learn_rew, context_trans_prob, cue_ambiguity, h  = lst

    if contingency_degradation and not switch_cues:
        fname = 'config_degradation_1_switch_0.json'
    elif contingency_degradation and switch_cues:
        fname = 'config_degradation_1_switch_1.json'
    elif not contingency_degradation and not switch_cues:
        fname = 'config_degradation_0_switch_0.json'
    elif not contingency_degradation and switch_cues:
        fname = 'config_degradation_0_switch_1.json'
    print(fname)


    file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)

    # try:
    #     file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)
    # except:
    #     create_trials(contingency_degradation=contingency_degradation, switch_cues=switch_cues)
    #     file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)


    task_params = js.load(file)                                                                                 

    colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
    sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
    starts = np.asarray(task_params['starts'])           # starting position of agent
    planets = np.asarray(task_params['planets'])         # planet positions 
    trial_type = np.asarray(task_params['trial_type'])
    blocks = np.asarray(task_params['block'])


    nblocks = int(blocks.max()+1)
    trials = blocks.size
    block = int(trials/nblocks)
 
    meta = {
        'trial_file' : fname, 
        'trial_type' : trial_type,
        'switch_cues': switch_cues == True,
        'contingency_degradation' : contingency_degradation == True,
        'learn_rew' : learn_rew == True,
        'context_trans_prob': context_trans_prob,
        'cue_ambiguity' : cue_ambiguity,
        'h' : h,
        'optimal_sequence' : sequence,
        'blocks' : blocks,
        'trials' : trials,
        'nblocks' : nblocks,
        'trials_per_block': block
    }

    all_optimal_seqs = np.unique(sequence)                                                                            

    # define reward probabilities dependent on time and position for reward generation process 
    Rho = np.zeros([trials, nr, ns])

    for i, pl in enumerate(planets):
        if i >= block*(nblocks-2) and i < block*(nblocks-1) and contingency_degradation:
            # print(i)
            # print(pl)
            Rho[i,:,:] = planet_reward_probs_switched[[pl]].T
        else:
            Rho[i,:,:] = planet_reward_probs[[pl]].T

    u = 0.99
    utility = np.array([(1-u)/2,(1-u)/2,u])


    reward_counts = planet_reward_probs.T*100
    # reward_counts = np.zeros(planet_reward_probs.shape) + 1

    reward_counts = np.ones([nr, npl, nc])
    par_list = [h,                        
                context_trans_prob,
                cue_ambiguity,            
                'avg',                    
                Rho,                      
                utility,                  
                state_transition_matrix,  
                planets,                  
                starts,                   
                colors,
                reward_counts,
                learn_rew]

    prefix = ''
    if switch_cues == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if contingency_degradation == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'

    worlds = [run_agent(par_list, trials, T, ns , na, nr, nc, npl) for _ in range(repetitions)]
    print('len worlds: ', len(worlds))
    worlds.append(meta)
    fname = prefix +'p' + str(cue_ambiguity) + '_q' + str(context_trans_prob) + '_h' + str(h) + '.json'
    fname = os.path.join(folder, fname)
    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(worlds)
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)
    
    print(fname)
    return fname


na = 2                                           # number of unique possible actions
nc = 4                                           # number of contexts, planning and habit
nr = 3                                           # number of rewards
ns = 6                                           # number of unique travel locations
npl = 3
steps = 3                                        # numbe of decisions made in an episode
T = steps + 1                                    # episode length

# reward probabiltiy vector
planet_reward_probs = np.array([[0.95, 0   , 0   ],
                                [0.05, 0.95, 0.05],
                                [0,    0.05, 0.95]]).T    # npl x nr

planet_reward_probs_switched = np.array([[0   , 0    , 0.95],
                                        [0.05, 0.95 , 0.05],
                                        [0.95, 0.05 , 0.0]]).T 

nplanets = 6
state_transition_matrix = np.zeros([ns,ns,na])

m = [1,2,3,4,5,0]
for r, row in enumerate(state_transition_matrix[:,:,0]):
    row[m[r]] = 1

j = np.array([5,4,5,6,2,2])-1
for r, row in enumerate(state_transition_matrix[:,:,1]):
    row[j[r]] = 1

state_transition_matrix = np.transpose(state_transition_matrix, axes= (1,0,2))
state_transition_matrix = np.repeat(state_transition_matrix[:,:,:,np.newaxis], repeats=nc, axis=3)


#######################

h =  [1]
cue_ambiguity = [0.8]
context_trans_prob = [nc]
degradation = [True]
cue_switch = [True]
learn_rew = [True]
arrays = [cue_switch, degradation, learn_rew, context_trans_prob, cue_ambiguity,h]
repetitions = 2
lst = []

for i in product(*arrays):
    lst.append(list(i))

names = []
for l in lst:
    name = run_single_sim(l, ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs, planet_reward_probs_switched,repetitions)
    names.append(name)

print(names)
















































# #%% 
# import pickle
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# #%%

# file = open("run_object.txt",'rb')
# run = pickle.load(file)

# file.close()


# trials = run.trials
# agent = run.agent
# perception = agent.perception
# environment= run.environment

# observations = run.observations
# true_optimal = environment.true_optimal
# context_cues = environment.context_cues
# trial_type = environment.trial_type
# policies = run.agent.policies
# actions = run.actions[:,:3] 
# executed_policy = np.zeros(trials)

# # 

# agent.posterior_context
# agent.posterior_rewards
# agent.posterior_policies
# agent.likelihood
# for pi, p in enumerate(policies):
#     inds = np.where( (actions[:,0] == p[0]) & (actions[:,1] == p[1]) & (actions[:,2] == p[2]) )[0]
#     executed_policy[inds] = pi
# reward = run.rewards


# data = pd.DataFrame({'executed': executed_policy,
#                      'optimal': true_optimal,
#                      'trial': np.arange(true_optimal.size),
#                      'trial_type': trial_type})

# data['chose_optimal'] = data.executed == data.optimal
# data['optimality'] = np.cumsum(data['chose_optimal'])/(data['trial']+1)
# fig = plt.figure()
# plt.subplot(1,2,1)
# ax = sns.scatterplot(data=data[['executed','chose_optimal']])
# cols = [[0,1,1], [1,0,0],[0,1,1]] 
# ranges = data.groupby('trial_type')['trial'].agg(['min', 'max'])
# for i, row in ranges.iterrows():
#     ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# plt.subplot(1,2,2)
# ax = sns.lineplot(data=data, x=data['trial'], y=data['optimality'])


# for i, row in ranges.iterrows():
#     ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# fig.figure.savefig('1.png', dpi=300) 
# # plt.close()

# #%%
# t = -1

# posterior_context = agent.posterior_context
# data = pd.DataFrame({'posterior_h3b': posterior_context[:t,3,0],
#                      'posterior_h6o': posterior_context[:t,3,1],
#                      'posterior_h6b': posterior_context[:t,3,2],
#                      'posterior_h3o': posterior_context[:t,3,3],
#                      'context_cue': context_cues[:t],    
#                      'trial_type': trial_type[:t],
#                      'trial': np.arange(trial_type[:t].size)})


# cols = [[0,1,1], [1,0,0],[0,1,1]] 
# fig2 = sns.lineplot(data=data.iloc[:,:4])
# ranges = data.groupby('trial_type')['trial'].agg(['min', 'max'])
# for i, row in ranges.iterrows():
#     fig2.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# # fig2 = sns.scatterplot(data=data.iloc[:,4])
# # fig2.figure.savefig('2.png',dpi=300)


# #%%

# posterior_rewards = agent.posterior_dirichlet_rew
# print(posterior_rewards.shape)

