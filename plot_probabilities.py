#%% 
import sys
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import chart_studio.plotly as py
# import plotly.express as px
import pandas as pd
# import cufflinks as cf
import json as js
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)
from itertools import product, repeat
import os
import action_selection as asl
from itertools import product
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import torch as ar
import perception as prc
import agent as agt
from environment import PlanetWorld
from agent import BayesianPlanner
from world import World
from planet_sequences import generate_trials_df
import time
import time
from multiprocessing import Pool
import multiprocessing.pool as mpp
import tqdm
from run_exampe_habit_v1 import run_agent

#%%
# def load_df(names,data_folder='data', extinguish=None):

#     if extinguish is None:
#         raise('did not specify if rewarded during extinction')

#     path = os.path.join(os.getcwd(),data_folder)
#     for fi, f in enumerate(names):
#         names[fi] = path + f

#     dfs = [None]*len(names)

#     for f,fname in enumerate(names):

#         # windows
#         if sys.platform == "win32":
#             fname = fname.replace('/','\\')
#         jsonpickle_numpy.register_handlers()
#         with open(fname, 'r') as infile:
#             data = json.load(infile)

#         worlds = pickle.decode(data)
#         meta = worlds[-1]

#         agents = [w.agent for w in worlds[:-1]]
#         perception = [w.agent.perception for w in worlds[:-1]]
#         nt = worlds[0].T
#         npl = perception[0].npl
#         nr = worlds[0].agent.nr
#         nc = perception[0].nc
#         nw = len(worlds[:-1])
#         ntrials = meta['trials']
#         learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
#         switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
#         contingency_degradation = np.repeat(meta['contingency_degradation'], ntrials*nw*nt)
#         tr_per_block = np.repeat(meta['trials_per_block'], ntrials*nw*nt)
#         ndb = np.repeat(meta['degradation_blocks'], ntrials*nw*nt)
#         ntb = np.repeat(meta['training_blocks'], ntrials*nw*nt)
#         utility_0 = np.repeat(prior_rewards[0], ntrials*nw*nt)
#         utility_1 = np.repeat(prior_rewards[1], ntrials*nw*nt)
#         utility_2 = np.repeat(prior_rewards[2], ntrials*nw*nt)
#         r_lambda = np.repeat(worlds[0].agent.perception.r_lambda, ntrials*nw*nt)
#         q = np.repeat(meta['context_trans_prob'], ntrials*nw*nt)
#         p = np.repeat(meta['cue_ambiguity'], ntrials*nw*nt)
#         h = np.repeat(meta['h'], ntrials*nw*nt)
#         dec_temp_cont = np.repeat(worlds[0].agent.perception.dec_temp_cont,ntrials*nw*nt)
#         dec_temp = np.repeat(worlds[0].dec_temp,ntrials*nw*nt)
#         switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
#         learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
#         trial_type = np.tile(np.repeat(meta['trial_type'], nt), nw)
#         trial = np.tile(np.repeat(np.arange(ntrials),nt), nw)
#         run = np.repeat(np.arange(nw),nt*ntrials)
#         run.astype('str')
#         agnt = np.repeat(np.arange(nw)+f*nw,nt*ntrials)


#         post_dir_rewards = [a.posterior_dirichlet_rew for a in agents]
#         post_dir_rewards = [post[:,1:,:,:] for post in post_dir_rewards]
#         entropy_rewards = np.zeros([nw*ntrials*nt,nc])

#         extinguished = np.zeros(ntrials*nw*nt, dtype='int32')
#         extinguished[:] = int(extinguish == True)
#         prior_rewards = worlds[0].agent.perception.prior_rewards

#         # Calculate reward entropy 
#         for ip, post in enumerate(post_dir_rewards):
#             post = post.sum(axis=3)                      # sum out planets
#             norm = post.sum(axis=2)                      # normalizing constants
#             reward_distributions = np.zeros(post.shape)

#             for r in range(nr):                          # normalize each reward 
#                 reward_distributions[:,:,r,:] = np.divide(post[:,:,r,:],norm)
#             entropy = np.zeros([ntrials, nt, nc])

#             for trl in range(ntrials):
#                 for t in range(nt-1):
#                     prob = reward_distributions[trl,t,:,:].T
#                     # if prob.sum() == 0:
#                     #     print('problem')
#                     entropy[trl,t+1,:] = -(np.log(prob)*prob).sum(axis=1)

#             entropy[:,0,:] = None
#             entropy_rewards[ip*(ntrials*nt):(ip+1)*(ntrials*nt),:] = np.reshape(entropy, [ntrials*nt,nc])

#         # calculate context entropy
#         entropy_context = np.zeros(ntrials*nt*nw)
#         post_context = [a.posterior_context for a in agents]

#         for ip, post in enumerate(post_context):
#             entropy = np.zeros([ntrials, nt])

#             for trl in range(ntrials):
#                 entropy[trl,:] = -(np.log(post[trl,:])*post[trl,:]).sum(axis=1) 
#             entropy_context[ip*(ntrials*nt):(ip+1)*(ntrials*nt)] = np.reshape(entropy, [ntrials*nt])
        
#         # Calculate choice optimality
#         posterior_context = [agent.posterior_context for agent in agents]
#         observations = [w.observations for w in worlds[:-1]]
#         context_cues = worlds[0].environment.context_cues
#         policies = worlds[0].agent.policies
#         actions = [w.actions[:,:3] for w in worlds[:-1]] 
#         true_optimal = np.tile(np.repeat(meta['optimal_sequence'],nt), nw)
#         cue = np.tile(np.repeat(context_cues, nt), nw)
#         ex_p = np.zeros(ntrials)
#         executed_policy = np.zeros(nw*ntrials,dtype='int32')
#         optimality = np.zeros(nw*ntrials)
#         chose_optimal = np.zeros(nw*ntrials)

#         for w in range(nw):
#             for pi, p in enumerate(policies):
#                 inds = np.where( (actions[w][:,0] == p[0]) & (actions[w][:,1] == p[1]) & (actions[w][:,2] == p[2]) )[0]
#                 ex_p[inds] = pi
#             executed_policy[w*ntrials:(w+1)*ntrials] = ex_p
#             ch_op = executed_policy[w*ntrials:(w+1)*ntrials] == meta['optimal_sequence']
#             chose_optimal[w*ntrials:(w+1)*ntrials] = ch_op
#             optimality[w*ntrials:(w+1)*ntrials] = np.cumsum(ch_op)/(np.arange(ntrials)+1)

#         executed_policy = np.repeat(executed_policy, nt)
#         chose_optimal = np.repeat(chose_optimal, nt)
#         optimality = np.repeat(optimality, nt)


#         # Calculate context optimality
#         no = perception[0].generative_model_context.shape[0]
#         # optimal_contexts = [np.argmax(perception[0].generative_model_contexts[i,:] for i in range(no))]
#         # degradation = np.repeat('contingency_degradation', ntrials*nw*nt)
#         inferred_context_t0 = np.zeros(ntrials*nw,dtype='int32')
#         inferred_context_t3  = np.zeros(ntrials*nw,'int32')
#         true_context = np.zeros(ntrials*nw, dtype='int32')
#         no, nc = perception[0].generative_model_context.shape 
#         modes_gmc =  perception[0].generative_model_context.argsort(axis=1)
#         contexts = [modes_gmc[i,:][-2:] for i in range(no)] # arranged in ascending order!
#         if_inferred_context_switch = np.zeros(ntrials, dtype="int32")

#         # c is an array holding what contexts should be for the given task trials [0,1,0,1,..2,3,2,3...0,1] etc
#         to = np.zeros(ntrials, dtype="int32")
#         for i in range(no):    
#             c = np.array([contexts[i][-1]]*(meta['trials_per_block']*meta['training_blocks'])\
#                 + [contexts[i][-2]]*(meta['trials_per_block']*meta['degradation_blocks'])\
#                 + [contexts[i][-1]]*meta['trials_per_block']*2)
#             to[np.where(context_cues == i)] = c[np.where(context_cues == i)]
#             # print(to)
#             if_inferred_context_switch[np.where(context_cues == i)] = c[np.where(context_cues == i)]
#         inferred_switch = np.zeros(ntrials*nw,dtype='int32')
#         context_optimality = np.zeros(ntrials*nw)

#         for w in range(nw):
#             inferred_context_t0[w*ntrials:(w+1)*ntrials] = np.argmax(posterior_context[w][:,0,:],axis=1)
#             inferred_context_t3[w*ntrials:(w+1)*ntrials] = np.argmax(posterior_context[w][:,-1,:],axis=1)
#             inferred_switch[w*ntrials:(w+1)*ntrials] = if_inferred_context_switch == \
#                                                     inferred_context_t3[w*ntrials:(w+1)*ntrials]
#             context_optimality[w*ntrials:(w+1)*ntrials] = np.cumsum(inferred_switch[w*ntrials:(w+1)*ntrials])\
#                                                                 /(np.arange(ntrials)+1)
#         true_context[w*ntrials:(w+1)*ntrials] = to
#         inferred_switch = np.repeat(inferred_switch, nt)
#         inferred_context_t0 = np.repeat(inferred_context_t0, nt)
#         inferred_context_t3 = np.repeat(inferred_context_t3, nt)
#         context_optimality = np.repeat(context_optimality, nt)
#         true_context =  np.repeat(true_context, nt)
#         t = np.tile(np.arange(4), nw*ntrials)
        
#         # print(true_context.size)
#         # print(trial_type.size)
#         d = {'trial_type':trial_type, 'run':run, 'trial':trial, 't':t, 'true_optimal':true_optimal,\
#                             'cue':cue, 'q':q, 'p':p, 'h':h, 'inferred_context_t0':inferred_context_t0,\
#                             'inferred_context_t3':inferred_context_t3, 'executed_policy':executed_policy,\
#                             'chose_optimal': chose_optimal, 'entropy_rew_c1': entropy_rewards[:,0], 'entropy_rew_c2': entropy_rewards[:,1], \
#                             'entropy_rew_c3': entropy_rewards[:,2] , 'entropy_rew_c4': entropy_rewards[:,3],\
#                             'policy_optimality':optimality,'agent':agnt, 'inferred_switch': inferred_switch,\
#                             'context_optimality':context_optimality, 'learn_rew': learn_rew, 'entropy_context':entropy_context, \
#                             'switch_cues':switch_cues, 'contingency_degradation': contingency_degradation,\
#                             'degradation_blocks': ndb, 'training_blocks':ntb, 'trials_per_block': tr_per_block,\
#                             'true_context': true_context, 'dec_temp':dec_temp, 'utility_0': utility_0, 'utility_1': utility_1, 'utility_2': utility_2,
#                             'r_lambda': r_lambda,'dec_temp_cont': dec_temp_cont} 

#         # for key in d.keys():
#         #     print(key, np.unique(d[key].shape))
#         dfs[f] = pd.DataFrame(d)
        
#     data = pd.concat(dfs)

#     groups = ['agent','run', 't','degradation_blocks', 'training_blocks', 'trials_per_block','learn_rew', 'p', 'q','h','cue']
#     grouped = data.groupby(by=groups)
#     data['iterator'] = 1
#     data['ith_cue_trial'] = grouped['iterator'].transform('cumsum')
#     data['policy_optimality_cue'] = grouped['chose_optimal'].transform('cumsum') / data['ith_cue_trial']
#     data['context_optimal_cue'] = grouped['inferred_switch'].transform('cumsum') / data['ith_cue_trial']
#     data.drop('iterator',1,inplace =True)
#     data.astype({'h': 'category'})

#     return data

#%%
nc = 4
extinguish = True


hs = [1,100]
cue_ambiguity = [0.8]                       
context_trans_prob = [0.85]
cue_switch = [False]
reward_naive = [True]
training_blocks = [6]
degradation_blocks=[6]
degradation = [True]
trials_per_block=[42]
dec_temps = [1,2]
rews = [0]
dec_temp_cont = [1,2,4]
utility = [[1, 9 , 90]]
conf = ['shuffled_and_blocked']


# name = 'multiple_hier_switch0_degr1_p0.8_learn_rew1_q0.85_h1_42_66_decp6_decc6_rew0_u1-9-90_shuffled_and_blocked_extinguish.json'
name = 'multiple_hier_switch0_degr1_p0.8_learn_rew1_q0.75_h1_42_66_decp1_decc1_rew0_u1-9-90_shuffled_and_blocked_extinguish.json'
fname = os.path.join('/'+conf[0] + '/' + name)
fname = fname.replace('/','\\')
# fname = name
names = [fname]
data_folder = 'temp'


# def load_df(names,data_folder='temp', extinguish=None):

path = os.path.join(os.getcwd(),data_folder)
for fi, f in enumerate(names):
    names[fi] = path + f

dfs = [None]*len(names)

for f,fname in enumerate(names):

    # windows
    if sys.platform == "win32":
        fname = fname.replace('/','\\')
    jsonpickle_numpy.register_handlers()
    with open(fname, 'r') as infile:
        data = json.load(infile)

    #agent data
    worlds = pickle.decode(data)
    meta = worlds[-1]
    worlds = worlds[:-1]
    agents = [w.agent for w in worlds]
    perceptions = [a.perception for a in agents]
    actions = [a.posterior_actions for a in agents]
    
    #constants
    nagents = len(worlds)
    ntrials = actions[0].shape[0]
    policies = agents[0].policies
    tt = actions[0].shape[1]
    nblocks = meta['nblocks']
    optimal_policy = agents[0].true_optimal
    
    # data frame entries shared across agents
    agnt = np.arange(nagents).repeat(ntrials)
    trial = np.tile(np.arange(ntrials),nagents)
    db = meta['degradation_blocks']
    tb = meta['training_blocks']
    tpb = meta['trials_per_block']
    trial_type = np.tile(np.array(meta['trial_type']), nagents)
    context_cues =np.tile(worlds[0].environment.context_cues,nagents)

    nth_trial = np.zeros(ntrials)
    nth_trial[:tb*tpb] = np.arange(tb*tpb)
    nth_trial[tb*tpb:(tb+db)*tpb] = np.arange(db*tpb)
    nth_trial[tb*tpb + db*tpb:] = np.arange((nblocks-(tb+db))*tpb)
    nth_trial = np.tile(nth_trial,nagents)
    
    # data frame entries calculated separately for each agent

    # Probability to select first action correctly
    action_probs = np.zeros([nagents*ntrials, 3])
    for tp in range(3):
        optimal_action_at_tp = policies[optimal_policy][:,tp]
        optimal_action_probability = np.zeros([nagents,ntrials])

        for ai, agent in enumerate(agents):
            for t in range(ntrials):
                optimal_action_probability[ai,t] = agent.posterior_actions[t,tp,optimal_action_at_tp[t]]
        
        action_probs[:,tp] = optimal_action_probability.reshape(nagents*ntrials)

    # Posterior over contexts

    for ai, agent in enumerate(agents):
        agent.posterior_context


    df_dict = {'agent':agnt, 'trial_type' : trial_type, 'trial':trial, 'nth_trial':nth_trial, 'context_cues':context_cues, 
               'prob_action_t0': action_probs[:,0], 'prob_action_t1': action_probs[:,1], 'prob_action_t2': action_probs[:,2]}
    df = pd.DataFrame(df_dict)
    df.groupby(['trial_type', 'context_cues']).mean('prob_action_t0')



# %%
conf = ['shuffled_and_blocked']
name = 'multiple_hier_switch0_degr1_p0.8_learn_rew1_q0.85_h100_42_66_decp1_decc1_rew0_u1-9-90_shuffled_and_blocked_extinguish.json'
fname = '/temp/'+ conf[0] + '/' + name
fname = (os.getcwd()+fname).replace('/','\\')

jsonpickle_numpy.register_handlers()

with open(fname, 'r') as infile:
    data = json.load(infile)

worlds = pickle.decode(data)
posterior_context = worlds[0].agent.posterior_context
trial_type = worlds[-1]['trial_type']
context_cue = worlds[0].environment.context_cues
tp = 1
cont0 = posterior_context[:,tp,0].round(5)
cont1 = posterior_context[:,tp,1].round(5)
cont2 = posterior_context[:,tp,2].round(5)
cont3 = posterior_context[:,tp,3].round(5)

df_dict = {'trial':np.arange(len(trial_type)), 'trial_type':trial_type, 'context_cue':context_cue,
           'c0':cont0,'c1':cont1,'c2':cont2,'c3':cont3}

df = pd.DataFrame(df_dict)
df.to_excel('test.xlsx')
# def categorize(row):
#     if row['trial_type'] == 0 and row['context_cue'] == 0:
#         return 0
#     elif row['trial_type'] == 0 and row['context_cue'] == 1:
#         return 1
#     elif row['trial_type'] == 1 and row['context_cue'] == 0:
#         return 2
#     elif row['trial_type'] == 1 and row['context_cue'] == 1:
#         return 3
#     elif row['trial_type'] == 2:
#         return -1

# df['true_context'] = df.apply(lambda row: categorize(row), axis=1)
# df['inferred_context'] = df[['c0', 'c1', 'c2','c3']].max(axis=1) == df['true_context'] 
# %%
