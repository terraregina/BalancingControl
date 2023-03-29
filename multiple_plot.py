import sys
import pickle
import os
import json
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from itertools import product, repeat

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch as ar

import matplotlib.gridspec as gridspec  
import string
import matplotlib.patheffects as pe
from sim_parameters import *


def load_file_names(arrays):
    lst = []
    for i in product(*arrays):
        lst.append(list(i))
    
    names = []
    print('files to load: ' + str(len(lst)))
    for li, l in enumerate(lst):


        prefix = '/multiple_'

        if use_fitting == True:
            prefix += 'fitt_'
        else:
            prefix +='hier_'

        if l[0] == True:
            prefix += 'switch1_'
        else:
            prefix +='switch0_'

        if l[1] == True:
            prefix += 'degr1_'
        else:
            prefix += 'degr0_'
        
        l[13] = [str(entry) for entry in l[13]]

        fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
        str(l[8]) + '_' + str(l[6]) + str(l[7]) + \
        '_decp' + str(l[9]) +'_decc' + str(l[10]) + '_rew' + str(l[11]) + '_' + 'u'+  '-'.join(l[13]) + '_'+ str(len(l[12])) + '_' + l[14] + '.json'

        names.append(fname)


    return names

def reshape(array,nt=4):
        shape = list(array[0].shape)
        shape = [shape[0]//nt, nt] + shape[1:]
        return [a.reshape(shape) for a in array]

def load_df(names,data_folder='data'):

    path = os.path.join(os.getcwd(),data_folder)
    #     names = os.listdir(path)
    for fi, f in enumerate(names):
        names[fi] = path + f
        # print(names[fi])

    dfs = [None]*len(names)

    for f,fname in enumerate(names):
        print(fname)
        # windows
        if sys.platform == "win32":
            fname = fname.replace('/','\\')
        jsonpickle_numpy.register_handlers()
        with open(fname, 'r') as infile:
            data = json.load(infile)
        worlds = pickle.decode(data)
        # worlds = worlds[-2:]\

        meta = worlds[-1]
        agents = [w.agent for w in worlds[:-1]]
        perception = [w.agent.perception for w in worlds[:-1]]
        nt = worlds[0].T
        npl = perception[0].npl
        nr = worlds[0].agent.nr
        nc = perception[0].nc
        nw = len(worlds[:-1])
        ntrials = meta['trials']
        learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
        switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
        contingency_degradation = np.repeat(meta['contingency_degradation'], ntrials*nw*nt)
        tr_per_block = np.repeat(meta['trials_per_block'], ntrials*nw*nt)
        ndb = np.repeat(meta['degradation_blocks'], ntrials*nw*nt)
        ntb = np.repeat(meta['training_blocks'], ntrials*nw*nt)

        if use_fitting:
            post_dir_rewards = [p.dirichlet_rew_params[1:,:,:,:,0] for p in perception]
            post_dir_rewards = reshape(post_dir_rewards, nt=nt-1)
        else:
            post_dir_rewards = [a.posterior_dirichlet_rew for a in agents]
            post_dir_rewards = [post[:,1:,:,:] for post in post_dir_rewards]
        entropy_rewards = np.zeros([nw*ntrials*nt,nc])
        prior_rewards = worlds[0].agent.perception.prior_rewards
        utility_0 = np.repeat(prior_rewards[0], ntrials*nw*nt)
        utility_1 = np.repeat(prior_rewards[1], ntrials*nw*nt)
        if nr != 2:
            utility_2 = np.repeat(prior_rewards[2], ntrials*nw*nt)
        else:
            
            utility_2 = np.repeat(False, ntrials*nw*nt)
        r_lambda = np.repeat(worlds[0].agent.perception.r_lambda, ntrials*nw*nt)

        for ip, post in enumerate(post_dir_rewards):
            post = post.sum(axis=3)                      # sum out planets
            norm = post.sum(axis=2)                      # normalizing constants
            reward_distributions = np.zeros(post.shape)

            for r in range(nr):                          # normalize each reward 
                reward_distributions[:,:,r,:] = np.divide(post[:,:,r,:],norm)
            entropy = np.zeros([ntrials, nt, nc])

            for trl in range(ntrials):
                for t in range(nt-1):
                    prob = reward_distributions[trl,t,:,:].T
                    # if prob.sum() == 0:
                    #     print('problem')
                    entropy[trl,t+1,:] = -(np.log(prob)*prob).sum(axis=1)

            entropy[:,0,:] = None
            entropy_rewards[ip*(ntrials*nt):(ip+1)*(ntrials*nt),:] = np.reshape(entropy, [ntrials*nt,nc])

        entropy_context = np.zeros(ntrials*nt*nw)
        if use_fitting:
            post_context = [p.posterior_contexts[...,0] for p in perception]
            post_context = reshape(post_context)
        else:
            post_context = [a.posterior_context for a in agents]

        for ip, post in enumerate(post_context):
            entropy = np.zeros([ntrials, nt])

            for trl in range(ntrials):
                entropy[trl,:] = -(np.log(post[trl,:])*post[trl,:]).sum(axis=1) 
            entropy_context[ip*(ntrials*nt):(ip+1)*(ntrials*nt)] = np.reshape(entropy, [ntrials*nt])

        # posterior_context = [agent.posterior_context for agent in agents]
        observations = [w.observations for w in worlds[:-1]]
        context_cues = worlds[0].environment.context_cues
        policies = worlds[0].agent.policies
        actions = [w.actions[:,:3] for w in worlds[:-1]] 
        true_optimal = np.tile(np.repeat(meta['optimal_sequence'],nt), nw)
        cue = np.tile(np.repeat(context_cues, nt), nw)
        executed_policy = np.zeros(nw*ntrials,dtype='int32')
        optimality = np.zeros(nw*ntrials)
        chose_optimal = np.zeros(nw*ntrials)

        for w in range(nw):
            executed_policy[w*ntrials:(w+1)*ntrials] = np.ravel_multi_index(actions[w].T,(2,2,2))
            chose_optimal[w*ntrials:(w+1)*ntrials] = executed_policy[w*ntrials:(w+1)*ntrials] == meta['optimal_sequence']
            optimality[w*ntrials:(w+1)*ntrials] = np.cumsum(chose_optimal[w*ntrials:(w+1)*ntrials])/(np.arange(ntrials)+1)

        executed_policy = np.repeat(executed_policy, nt)
        chose_optimal = np.repeat(chose_optimal, nt)
        optimality = np.repeat(optimality, nt)
        q = np.repeat(meta['context_trans_prob'], ntrials*nw*nt)
        p = np.repeat(meta['cue_ambiguity'], ntrials*nw*nt)
        h = np.repeat(meta['h'], ntrials*nw*nt)
        dec_temp_cont = np.repeat(worlds[0].agent.perception.dec_temp_cont,ntrials*nw*nt)
        dec_temp = np.repeat(worlds[0].dec_temp,ntrials*nw*nt)
        switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
        learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
        degradation = np.repeat('contingency_degradation', ntrials*nw*nt)
        trial_type = np.tile(np.repeat(meta['trial_type'], nt), nw)
        trial = np.tile(np.repeat(np.arange(ntrials),nt), nw)
        run = np.repeat(np.arange(nw),nt*ntrials)
        run.astype('str')
        agnt = np.repeat(np.arange(nw)+f*nw,nt*ntrials)

        true_context = np.zeros(ntrials*nw, dtype='int32')
        no, nc = perception[0].generative_model_context.shape 
        
        modes_gmc =  perception[0].generative_model_context.argsort(axis=1)
        contexts = np.array([modes_gmc[i,:][-2:] for i in range(no)]) # arranged in ascending order!
        contexts.sort()
        true_cont_short = np.zeros(ntrials, dtype="int32")

        context_optimality = np.zeros(ntrials*nw)
        context_mapping = np.zeros([np.unique(trial_type).size, no])
        context_mapping[[0,2],:] = [0,1]
        context_mapping[1,:] = [2,3]
        true_context = context_mapping[trial_type[:ntrials*nt], cue[:ntrials*nt]]
        true_cont_short = context_mapping[trial_type[np.arange(0,ntrials*nt, 4)], cue[np.arange(0, ntrials*nt, 4)]]

        cs = np.zeros([ntrials*nw, nc],'int32')
        switch = np.zeros([ntrials*nw, nc], 'int32')

        for w in range(nw):
            for i in range(nt):
                cs[w*ntrials:(w+1)*ntrials,i] = np.argmax(post_context[w][:,i,:],axis=1)

                switch[w*ntrials:(w+1)*ntrials,i] = true_cont_short == \
                                                        cs[w*ntrials:(w+1)*ntrials,i]
            
            inferred_switch = np.logical_and(np.logical_and(np.logical_and(switch[:,0], switch[:,1]), switch[:,2]) ,switch[:,3])
            context_optimality[w*ntrials:(w+1)*ntrials] = np.cumsum(inferred_switch[w*ntrials:(w+1)*ntrials])\
                                                                /(np.arange(ntrials)+1)
        inferred_switch = np.repeat(inferred_switch, nt)
        inferred_context = cs
        inferred_context = inferred_context.reshape(inferred_context.size)

        context_optimality = np.repeat(context_optimality, nt)
        true_context =  np.tile(true_context,nw)
        exp_reward = np.tile(np.repeat(meta['exp_reward'],nt),nw)
        t = np.tile(np.arange(4), nw*ntrials)


        d = {\
        'agent':agnt, 'run':run, 'trial':trial, 't':t, 'trial_type':trial_type,'cue':cue,
        'true_context': true_context,\
        'inferred_context': inferred_context,'inferred_switch': inferred_switch, 'context_optimality':context_optimality,
        'true_optimal':true_optimal, 'executed_policy':executed_policy,'chose_optimal': chose_optimal,\
        'policy_optimality':optimality, 'cont_reward':exp_reward,\
        'pol_reward':exp_reward, 'h':h,\
        'entropy_rew_c1': entropy_rewards[:,0], 'entropy_rew_c2': entropy_rewards[:,1], \
        'entropy_rew_c3': entropy_rewards[:,2], 'entropy_rew_c4': entropy_rewards[:,3],\
        'learn_rew': learn_rew, 'entropy_context':entropy_context, \
        'switch_cues':switch_cues, 'contingency_degradation': contingency_degradation,\
        'degradation_blocks': ndb, 'training_blocks':ntb, 'trials_per_block': tr_per_block,\
        'dec_temp':dec_temp, 'utility_0': utility_0, 'utility_1': utility_1, 'utility_2': utility_2,
        'r_lambda': r_lambda,'dec_temp_cont': dec_temp_cont, 'q':q, 'p':p} 

        dfs[f] = pd.DataFrame(d)
        
    data = pd.concat(dfs)

    groups = ['agent','run', 't','degradation_blocks', 'training_blocks', 'trials_per_block','learn_rew', 'p', 'q','h','cue']
    grouped = data.groupby(by=groups)
    data['iterator'] = 1
    data['ith_cue_trial'] = grouped['iterator'].transform('cumsum')
    data['policy_optimality_cue'] = grouped['chose_optimal'].transform('cumsum') / data['ith_cue_trial']
    data['context_optimal_cue'] = grouped['inferred_switch'].transform('cumsum') / data['ith_cue_trial']
    data.drop('iterator',1,inplace =True)
    data.astype({'h': 'category'})

    return data
5

def load_df_animation_context(names,data_folder='temp'):

    path = os.path.join(os.getcwd(),data_folder)
    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)


    overall_df = [None for _ in range(len(names))]
    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            data = json.load(infile)

        worlds = pickle.decode(data)
        meta = worlds[-1]
        nw = len(worlds[:-1])
        agents = [w.agent for w in worlds[:-1]]
        posterior_context = [agent.posterior_context for agent in agents]
        ntrials, t, nc = posterior_context[0].shape
        outcome_surprise = [agent.outcome_suprise for agent in agents]
        policy_surprise = [agent.policy_surprise for agent in agents]
        policy_entropy = [agent.policy_entropy for agent in agents]
        npi = agents[0].posterior_policies.shape[2]
        taus = np.arange(ntrials)
        ts = np.arange(t)
        cs = np.arange(nc)
        pis_post = np.array(['post_' for _ in range(npi)], dtype=object) + np.array([str(i) for i in range(npi)], dtype=object)
        pis_prior = np.array(['post_' for _ in range(npi)], dtype=object) + np.array([str(i) for i in range(npi)], dtype=object)
        pis_like = np.array(['post_' for _ in range(npi)], dtype=object) + np.array([str(i) for i in range(npi)], dtype=object)

        mi = pd.MultiIndex.from_product([taus, ts, cs], names=['trial', 't', 'context'])
        mi_post = pd.MultiIndex.from_product([taus, ts, pis_post, cs], names=['trial', 't', 'policy','context'])
        mi_like = pd.MultiIndex.from_product([taus, ts, pis_like, cs], names=['trial', 't', 'policy','context'])
        mi_prior = pd.MultiIndex.from_product([taus, ts, pis_like, cs], names=['trial', 't', 'policy','context'])

        dfs = [None for _ in range(nw)]
        factor = ntrials*nc*t

        for w in range(nw):

            policy_post_df =  pd.Series(index=mi_post, data=agents[w].posterior_policies.flatten())
            policy_post_df = policy_post_df.unstack(level='policy').reset_index()
            # fix entropy calcs
            policy_prob = policy_post_df.iloc[:,-8:].to_numpy()
            policy_prob[policy_prob == 0] = 10**(-300)
            entropy = -(policy_prob*np.log(policy_prob)).sum(axis=1)

            prior_policies = np.tile(agents[w].prior_policies[:,np.newaxis,:,:], (1,t,1,1))
            # print(np.all(prior_policies[:,0,:,:] == prior_policies[:,1,:,:] ))
            policy_prior_df =  pd.Series(index=mi_prior, data=prior_policies.flatten())
            policy_prior_df = policy_prior_df.unstack(level='policy').reset_index()

            policy_like_df =  pd.Series(index=mi_like, data=agents[w].likelihood.flatten())
            policy_like_df = policy_like_df.unstack(level='policy').reset_index()
            
            policy_entropy_df = pd.Series(index=mi,\
                data=policy_entropy[w].flatten()).reset_index().rename(columns = {0:'policy_entropy'})
            
            policy_surprise_df = pd.Series(index=mi,\
                data=policy_surprise[w].flatten()).reset_index().rename(columns = {0:'context_obs_surprise'})
            outcome_surprise_df = pd.Series(index=mi,\
                data=outcome_surprise[w].flatten()).reset_index().rename(columns = {0:'outcome_surprise'})
                
            # break
            df = pd.Series(index=mi, data=posterior_context[w].flatten())
            df = df.reset_index().rename(columns = {0:'probability'})
            df['context_obs_surprise'] = policy_surprise_df['context_obs_surprise']
            df['outcome_surprise'] = outcome_surprise_df['outcome_surprise']
            df['policy_entropy'] = policy_entropy_df['policy_entropy']
            # df['policy_entropy'] = policy_entropy_df['policy_entropy']
            df['learn_rew'] = np.repeat(meta['learn_rew'], factor)
            df['switch_cues'] = np.repeat(meta['switch_cues'], factor)
            df['contingency_degradation'] = np.repeat(meta['contingency_degradation'], factor)
            df['trials_per_block'] = np.repeat(meta['trials_per_block'], factor)
            df['degradation_blocks'] = np.repeat(meta['degradation_blocks'], factor)
            df['training_blocks'] = np.repeat(meta['training_blocks'], factor)
            df['context_cues'] = np.repeat(worlds[0].environment.context_cues, nc*t)
            df['true_optimal'] = np.repeat(meta['optimal_sequence'], nc*t)
            df['q'] = np.repeat(meta['context_trans_prob'], factor)
            df['p'] = np.repeat(meta['cue_ambiguity'], factor)
            df['h'] = np.repeat(meta['h'], factor)
            df['run'] = np.repeat(w,factor)
            df['trial_type'] = np.repeat(meta['trial_type'], nc*t)
            df['trial'] = np.repeat(np.arange(ntrials), nc*t)
            df['agent'] = np.repeat(w+f*nw,factor)
            df = df.join(policy_post_df.iloc[:,-8:])
            dfs[w] = df
            break
        overall_df[f] = pd.concat(dfs)
    data_animation = pd.concat(overall_df)
    # data_animation.to_excel('data_animation.xlsx')
    return data_animation
# data_animation.to_csv('data_animation.csv')

def load_df_animation_pol(names,data_folder='temp'):

    path = os.path.join(os.getcwd(),data_folder)
    #     names = os.listdir(path)

    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)


    overall_df = [None for _ in range(len(names))]
    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            data = json.load(infile)

        worlds = pickle.decode(data)
        meta = worlds[-1]
        nw = len(worlds[:-1])
        agents = [w.agent for w in worlds[:-1]]
        posterior_context = [agent.posterior_context for agent in agents]
        ntrials, t, nc = posterior_context[0].shape
        policy_entropy = [agent.policy_entropy for agent in agents]
        npi = agents[0].posterior_policies.shape[2]
        taus = np.arange(ntrials)
        ts = np.arange(t)
        cs = np.arange(nc)
        pis = np.arange(npi)

        mi = pd.MultiIndex.from_product([taus, ts, pis, cs], names=['trial', 't','policy', 'context'])
        dfs = [None for _ in range(nw)]
        factor = ntrials*nc*t*npi

        for w in range(nw):
            df =  pd.Series(index=mi, data=agents[w].posterior_policies.flatten())
            df = df.reset_index().rename(columns = {0:'policy_post'})

            prior_policies = np.tile(agents[w].prior_policies[:,np.newaxis,:,:], (1,t,1,1))
            policy_prior_df =  pd.Series(index=mi, data=prior_policies.flatten())
            policy_prior_df = policy_prior_df.reset_index().rename(columns = {0:'policy_prior'})

            policy_like_df =  pd.Series(index=mi, data=agents[w].likelihood.flatten())
            policy_like_df = policy_like_df.reset_index().rename(columns = {0:'policy_likelihood'})
            df['policy_prior'] = policy_prior_df['policy_prior']
            df['policy_like'] = policy_like_df['policy_likelihood']
            post_context = np.tile(posterior_context[w][:,:,np.newaxis,:], (1,1,npi,1))
            # print(np.all(post_context[:,:,0,:] == post_context[:,:,3,:]))

            post_context_df = pd.Series(index=mi, data=post_context.flatten())
            post_context_df = post_context_df.reset_index().rename(columns = {0:'post_context'})
            df['post_context'] = post_context_df['post_context']
            df['learn_rew'] = np.repeat(meta['learn_rew'], factor)
            df['switch_cues'] = np.repeat(meta['switch_cues'], factor)
            df['contingency_degradation'] = np.repeat(meta['contingency_degradation'], factor)
            df['trials_per_block'] = np.repeat(meta['trials_per_block'], factor)
            df['degradation_blocks'] = np.repeat(meta['degradation_blocks'], factor)
            df['training_blocks'] = np.repeat(meta['training_blocks'], factor)
            df['context_cues'] = np.repeat(worlds[0].environment.context_cues, nc*t*npi)
            df['true_optimal'] = np.repeat(meta['optimal_sequence'], nc*t*npi)
            df['q'] = np.repeat(meta['context_trans_prob'], factor)
            df['p'] = np.repeat(meta['cue_ambiguity'], factor)
            df['h'] = np.repeat(meta['h'], factor)
            df['run'] = np.repeat(w,factor)
            df['trial_type'] = np.repeat(meta['trial_type'], nc*t*npi)
            df['trial'] = np.repeat(np.arange(ntrials), nc*t*npi)
            df['agent'] = np.repeat(w+f*nw,factor)
            dfs[w] = df

        overall_df[f] = pd.concat(dfs)
    data_animation = pd.concat(overall_df)
    # data_animation.to_excel('data_animation.xlsx')
    return data_animation

def load_df_reward_dkl(names,planet_reward_probs, planet_reward_probs_switched,data_folder='temp',nc=4):

    path = os.path.join(os.getcwd(),data_folder)
    #     names = os.listdir(path)

    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)

    dfs = [None]*len(names)

    overall_df = [None for _ in range(len(names))]

    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()
        with open(fname, 'r') as infile:
            data = json.load(infile)
                         
        worlds = pickle.decode(data)
        meta = worlds[-1]
        agents = [w.agent for w in worlds[:-1]]
        perception = [w.agent.perception for w in worlds[:-1]]

        if use_fitting:
            reward_probs = [p.dirichlet_rew_params[1:,:,:,:,0] for p in perception]
            reward_probs = reshape(reward_probs,3)
            reward_probs = [np.insert(probs,0,0,axis=1) for probs in reward_probs]
        else:
            reward_probs = [a.posterior_dirichlet_rew for a in agents]
        
        prior_rewards = worlds[0].agent.perception.prior_rewards

        nt = worlds[0].T
        npl = perception[0].npl
        nr = worlds[0].agent.nr
        nc = perception[0].nc
        nw = len(worlds[:-1])
        ntrials = meta['trials']
        
        # define true distribution reward
        tpb = meta['trials_per_block']
        db = meta['degradation_blocks']
        tb = meta['training_blocks']
        # p= np.tile(planet_reward_probs[np.newaxis,np.newaxis,:,:,:], (ntrials, nt, 1,1,1))
        # p[tb*tpb:(tb+db)*tpb,:,:,:,:] = \
        #     np.tile(planet_reward_probs_switched[np.newaxis,np.newaxis,:,:,:],
        #             ((db + tb)*tpb - tb*tpb, nt, 1,1,1))

        p = np.tile(planet_reward_probs[None, None, :,:,None], (ntrials, nt, 1,1,nc))
        p[tb*tpb:(tb+db)*tpb] = np.tile(planet_reward_probs_switched[None, None, :,:,None],\
                                     (db*tpb, nt, 1,1,nc))
        p[p == 0] = 10**(-300)
        # else:
        #     pass

        factor = ntrials*nt*nc
        taus = np.arange(ntrials)
        ts = np.arange(nt)
        # npls = np.char.add(np.asarray(['pl_' for _ in range(npl)]), np.asarray([str(i) for i in range(npl)]))
        npls = np.arange(npl)
        nrs = np.arange(nr)
        cs = np.arange(nc)
        mi = pd.MultiIndex.from_product([taus, ts, npls, cs],
                names=['trial', 't', 'planet', 'context'])
        # dfs_dkl = [None for _ in range(nw)]
        # factor = ntrials*nt*nr*npl*nc

        dkl_df = [None for _ in range(nw)]
        for w in range(nw):
            q = reward_probs[w]                         
            e = (db+tb)*tpb                             
            # q[e:,:,:,:,:] = np.tile(q[e-1,:,:,:,:], (2*tpb,1,1,1,1)) #why would I do this? 
            q[e:,:,:,:,:] = np.tile(q[e-1,:,:,:,:], (tpb,1,1,1,1)) #why would I do this? 

            q[q == 0] = 10**(-300)
            norm = 1/(q.sum(axis=2))                    # transform counts to distributions
            q = np.einsum('etrpc, etpc -> etrpc', q, norm)
            dkl = (q*np.log(q/p)).sum(axis=2)
            df =  pd.Series(index=mi, data=dkl.flatten())
            df = df.unstack(level = 'planet')
            df = df.reset_index()
            df['avg_dkl'] = df.iloc[:,-2:].sum(axis=1)/nr

            df['learn_rew'] = np.repeat(meta['learn_rew'], factor)
            df['switch_cues'] = np.repeat(meta['switch_cues'], factor)
            df['contingency_degradation'] = np.repeat(meta['contingency_degradation'], factor)
            df['trials_per_block'] = np.repeat(meta['trials_per_block'], factor)
            df['degradation_blocks'] = np.repeat(meta['degradation_blocks'], factor)
            df['training_blocks'] = np.repeat(meta['training_blocks'], factor)
            df['context_cues'] = np.repeat(worlds[0].environment.context_cues, nc*nt)
            df['true_optimal'] = np.repeat(meta['optimal_sequence'], nc*nt)
            df['q'] = np.repeat(meta['context_trans_prob'], factor)
            df['p'] = np.repeat(meta['cue_ambiguity'], factor)
            df['dec_temp'] = np.repeat(worlds[0].dec_temp,factor)
            df['dec_temp_cont'] = np.repeat(worlds[0].agent.perception.dec_temp_cont,factor)

            df['h'] = np.repeat(meta['h'], factor)
            df['run'] = np.repeat(w,factor)
            df['trial_type'] = np.repeat(meta['trial_type'], nc*nt)
            df['trial'] = np.repeat(np.arange(ntrials), nc*nt)
            df['agent'] = np.repeat(w+f*nw,factor)
            df['utility_0'] = np.repeat(prior_rewards[0], factor)
            df['utility_1'] = np.repeat(prior_rewards[1], factor)

            if nr == 3:
                df['utility_2'] = np.repeat(prior_rewards[2], factor)
            else:
                df['utility_2'] = np.repeat(False, factor)

            df['r_lambda'] = np.repeat(worlds[0].agent.perception.r_lambda, factor)
            dkl_df[w] = df
        #     break
        # break
        overall_df[f] = pd.concat(dkl_df)
    data = pd.concat(overall_df)
    return data



def plot_all(lst,hs=[[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]],utility=[[1,9,90]], testing=False):

    for l in lst:
        names_arrays = [[l[0]], [l[1]], [l[2]], [l[3]], [l[4]], hs,\
                [l[5]], [l[6]], [l[7]],[l[8]],[l[9]],[l[10]],[l[11]], utility, [l[12]] ] 
        data_folder = 'temp/'+l[12]
        names = load_file_names(names_arrays)
        df = load_df(names, data_folder=data_folder)
        df_dkl = load_df_reward_dkl(names, planet_reward_probs, planet_reward_probs_switched,\
                                    data_folder=data_folder)
        # df.to_excel('test.xlsx')
        
        df.head()


        switch = l[0]
        contingency_degr = l[1]
        reward_naive = l[2]

        h = 200
        q = l[3]
        p = l[4]
        t = 3
        training_blocks = l[5]
        degradation_blocks = l[6]
        db = l[6]
        trials_per_block = l[7]
        dec_temp = l[8]
        dec_temp_cont = l[9]
        rew = l[10]
        rewards = l[11]
        nr = len(rewards)
        cue = 0
        one_run = False
        nr = len(rewards)
        queries =  ['p==' + str(p)]
        sns.set_style("darkgrid")

        palette = ['Blues_r','Reds_r']
        titles = ['Habit', 'Planning']

        title_pad = 18
        title_fs = 12

        x_pad=10
        x_fs=10

        y_pad=12
        y_fs=12


        for util in utility:
            
            util = [u/100 for u in util]

            if len(util) == 2:
                util.append(False)

            # define whole figure
            fig = plt.figure(figsize=(15, 18))
            gs0 = gridspec.GridSpec(3, 1, figure=fig, hspace=0.6, height_ratios=[3, 1, 1.3])
            gs01 = gs0[0].subgridspec(2, 3,hspace=0.4, wspace=0.3)
            gs02 = gs0[2].subgridspec(1,4, wspace=0.35)

            ########## plot accuracy ########### 

            strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=', '& p==',\
                            '& training_blocks==', '& degradation_blocks==', '& trials_per_block==',\
                             '& dec_temp ==',\
                            '& dec_temp_cont ==', '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
                            
            vals = np.array([switch, contingency_degr, reward_naive, q, h, p, \
                            training_blocks, db, trials_per_block, dec_temp, \
                            dec_temp_cont,\
                            util[0], util[1], util[2],rew], dtype='str')
            whole_query = np.char.join('', np.char.add(strs, vals))

            base_query = ' '.join(whole_query.tolist())
            if one_run == True:
                base_query += ' & run == 0'

            base_df = df.query(base_query)
            base_df = base_df.astype({'h': 'category'})
            plot_df = base_df.query('t==0 & degradation_blocks == ' + str(degradation_blocks))
            grouped = plot_df.groupby(by=['agent', 'run','h','trial_type', 'cue'])
            plot_df['policy_optimality_subset'] = grouped['chose_optimal'].transform('cumsum')
            plot_df['offset'] = grouped['ith_cue_trial'].transform('min')
            # plot_df['ith_cue_trial'] =
            plot_df['policy_optimality_subset'] = plot_df['policy_optimality_subset'] / ( plot_df['ith_cue_trial'] - plot_df['offset']+1)

            palette = ['Blues_r','Reds_r']
            titles = ['Training', 'Degradation', 'Extinction']

            # axes = np.array([[plt.subplot(grid[1, :4]), plt.subplot(grid[1, 4:8]), plt.subplot(grid[1, 8:12])],\
            #                 [plt.subplot(grid[2, :4]), plt.subplot(grid[2, 4:8]), plt.subplot(grid[2, 8:12])]])

            axes = np.array([[fig.add_subplot(gs01[0,0]), fig.add_subplot(gs01[0,1]), fig.add_subplot(gs01[0,2])],
                             [fig.add_subplot(gs01[1,0]), fig.add_subplot(gs01[1,1]), fig.add_subplot(gs01[1,2])]])
            ax0 = axes[0,0]
            x_titles = ['Habit Trial', 'Planning Trial']
            # base_df[base_df.columns[:18]].to_excel('test.xlsx')
            for phase in [0,1,2]:
                for cue in [0,1]:
                    pal = sns.color_palette(palette[cue],n_colors=np.unique(plot_df['h']).size)
                    sns.lineplot(ax = axes[cue,phase], data=plot_df.query('trial_type ==' + str(phase) + '& cue ==' + str(cue)),\
                                x = 'ith_cue_trial',y='policy_optimality_subset', hue='h', legend=False,\
                                palette=pal, ci='sd')

                    hex = pal.as_hex()
                    
                    lx, ly, ux, uy = axes[cue,phase].get_position(original=True).bounds

                    if phase == 0:
                        axes[cue,phase].text(ux*2.0,uy*2, 'Strong Habit Learner',\
                                             transform=axes[cue,phase].transAxes,\
                                             color=hex[0], fontsize=12)

                        tl = axes[cue,phase].text(ux*2.0,uy*.8, 'Weak Habit Learner',\
                                             transform=axes[cue,phase].transAxes,\
                                             color=hex[-1], fontsize=12)

                        tl.set_path_effects([pe.PathPatchEffect(offset=(0.8, -0.7),
                                                    edgecolor='black', linewidth=0.1,
                                                    facecolor='black'),
                        pe.PathPatchEffect(edgecolor=hex[-1], linewidth=0.1,
                                                    facecolor=hex[-1])])

                    axes[cue,phase].set_xlabel(x_titles[cue],labelpad=10,fontsize=10)

                axes[0,phase].set_title(titles[phase],fontsize=title_fs,pad=title_pad, weight='bold')

                ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
                cols = [[1,1,1], [0,0,0],[1,1,1]] 


                for n, ax in enumerate(axes.flatten()):
                    ax.set_ylim([0,1])
                    # ax.set_xlabel('trial')
                    if n < 3:
                        ax.set_ylabel('Habit policy optimality', fontsize=y_fs, labelpad=y_pad)
                    else:
                        ax.set_ylabel('Planning policy optimality', fontsize=y_fs, labelpad=y_pad)

            ########## plot context ########### 
            
            gs00 = gs0[1].subgridspec(1, 2,wspace=0.3)
            axes = np.array([fig.add_subplot(gs00[:, 0]), fig.add_subplot(gs00[:, 1])])
            ax1 = axes[0]

            strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
                             '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
                             '& dec_temp_cont ==',
                            '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
                            
            vals = np.array([switch, contingency_degr, reward_naive, q, h, \
                            training_blocks, db, trials_per_block, dec_temp,
                              dec_temp_cont,\
                            util[0], util[1], util[2],rew], dtype='str')

            whole_query = np.char.join('', np.char.add(strs, vals))
            base_query = ' '.join(whole_query.tolist())

            if one_run == True:
                base_query += ' & run == 0'

            base_df = df.query(base_query)


            lgnd = [False, True]
            cues = [0,1]
            titles_c = ['Habit cue','Planning Cue']


            query ='p == ' + str(p) 
            t = 3
            for cue in cues:

               plot_df = base_df.query(query + '& t ==' + str(t) + ' & cue == ' + str(cue))

               pal = sns.color_palette(palette[cue],n_colors=np.unique(plot_df['h']).size)
               sns.lineplot(ax = axes[cue], data=plot_df, x='ith_cue_trial', y='context_optimal_cue', hue='h',\
                            palette=pal,legend=False, ci='sd')
               axes[cue].set_title(titles_c[cue],fontsize=title_fs, pad=title_pad, weight='bold')\
                   
            
               hex = pal.as_hex()
    
                
               lx, ly, ux, uy = axes[cue].get_position(original=True).bounds
    
    
               axes[cue].text(ux*1.84,uy*3, 'Strong Habit Learner',\
                                        transform=axes[cue].transAxes,\
                                        color=hex[0], fontsize=12)
    
               tl = axes[cue].text(ux*1.84,uy*1.05, 'Weak Habit Learner',\
                                        transform=axes[cue].transAxes,\
                                        color=hex[-1], fontsize=12)
               tl.set_path_effects([pe.PathPatchEffect(offset=(0.8, -0.7),
                                            edgecolor='black', linewidth=0.1,
                                            facecolor='black'),
                pe.PathPatchEffect(edgecolor=hex[-1], linewidth=0.1,
                                                    facecolor=hex[-1])])
            ranges = plot_df.groupby('trial_type')['ith_cue_trial'].agg(['min', 'max'])

            cols = [[1,1,1], [0,0,0],[1,1,1]]
            for ax in axes.flatten():
                ax.set_ylim([0,1.05])
                for i, row in ranges.iterrows():
                    ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
                    ax.set_ylabel('Context optimality',fontsize=y_fs, labelpad=y_pad)
                    
            axes[0].set_xlabel(x_titles[0],fontsize=x_fs, labelpad=x_pad)
            axes[1].set_xlabel(x_titles[1],fontsize=x_fs, labelpad=x_pad)
            

            ########## plot dkl ########### 

            axes = np.array([fig.add_subplot(gs02[0,0]), fig.add_subplot(gs02[0,1]),\
                             fig.add_subplot(gs02[0,2]), fig.add_subplot(gs02[0,3])])
            ax2 = axes[0]

            strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
                             '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
                             '& dec_temp_cont ==',
                             '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
                            
            vals = np.array([switch, contingency_degr, reward_naive, q, h, \
                            training_blocks, db, trials_per_block, dec_temp,\
                              dec_temp_cont,\
                            util[0], util[1], util[2],rew], dtype='str')

            whole_query = np.char.join('', np.char.add(strs, vals))
            base_query = ' '.join(whole_query.tolist())
            base_df_dkl = df_dkl.query(base_query)
            plot_df = base_df_dkl.query('t==3' + ' & degradation_blocks ==' + str(db))

            palette = palette + palette
            titles = ['Training Habit context','Training Planning context','Degradation Habit context','Degradation Planning context']
            for cont in range(4):
                if one_run == True:
                    quer = 'context== ' + str(cont) + ' & run == 0'
                else:
                    quer = 'context== ' + str(cont)

                pal = sns.color_palette(palette[cont],n_colors=np.unique(plot_df['h']).size)

                sns.lineplot(ax=axes[cont], data=plot_df.query(quer),
                            x='trial', y='avg_dkl', hue='h', \
                            legend=False, \
                            palette=pal, ci='sd')
                axes[cont].set_title(titles[cont],fontsize=title_fs, pad=title_pad, weight="bold")


                hex = pal.as_hex()
                
                if cont <2:
                    lx, ly, ux, uy = axes[cont].get_position(original=True).bounds


                    axes[cont].text(ux*0.5, uy*6.7, 'Strong Habit Learner',\
                                            transform=axes[cont].transAxes,\
                                            color=hex[0], fontsize=12)
                    t = axes[cont].text(ux*0.5, uy*5.7, 'Weak Habit Learner',\
                                            transform=axes[cont].transAxes,\
                                            color=hex[-1], fontsize=12)
                    t.set_path_effects([pe.PathPatchEffect(offset=(0.8, -0.7),
                                                  edgecolor='black', linewidth=0.1,
                                                 facecolor='black'),
                    pe.PathPatchEffect(edgecolor=hex[-1], linewidth=0.1,
                                                 facecolor=hex[-1])])


            ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
            cols = [[1,1,1], [0,0,0],[1,1,1]]

            ylim_dkl = plot_df['avg_dkl'].max()*1.7
            for ax in axes.flatten():
                ax.set_ylim([0,ylim_dkl])    
                for i, row in ranges.iterrows():
                    ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
                    ax.set_ylabel('Average DKL',fontsize=y_fs, labelpad=5)
                    ax.set_xlabel('Trial', fontsize=x_fs, labelpad = x_pad)

            fname = 'figs/'+ l[-1] + '_'
            if use_fitting:
                agent_type = 'fitt'
            else:
                agent_type = 'hier'

            if testing:
                fname += '_'.join(['multiple',agent_type,'switch', str(int(l[0])), 'degr', str(int(l[1])),\
                    'learn', str(int(l[2])), 'q', str(l[3]),\
                    'p', str(l[4]),'dec', str(l[8]),str(training_blocks) + str(degradation_blocks), 'util', '-'.join([str(u) for u in util]), ''.join([str(hhh) for hhh in hs])]) + '.png'
            else:
                fname += '_'.join(['multiple',agent_type,'switch', str(int(l[0])), 'degr', str(int(l[1])),\
                        'learn', str(int(l[2])), 'q', str(l[3]),\
                        'p', str(l[4]),'decp', str(l[8]),'decc', str(l[9]),\
                        str(training_blocks) + str(degradation_blocks), 'util', '-'.join([str(u) for u in util])]) + '_nr_' + str(nr) +'.png'
            
            
            #            fnames = os.path.join(os.getcwd(), fname)
            fnames = os.getcwd() + '/' + fname
            # windows
            if sys.platform == 'win32':
                fnames = fnames.replace('/','\\')
            print(fname)

            ratio = [0.333, 0.5, 0.22]

            for n, ax in enumerate([ax0, ax1, ax2]):
                ax.text(-0.11/ratio[n], 1.3, string.ascii_uppercase[n], transform=ax.transAxes, 
                        size=18, weight='bold')
            
  
            fig.savefig(fnames, dpi=100)    
            # fig.savefig(fnames, dpi=300)    



arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,\
        training_blocks, degradation_blocks, trials_per_block, dec_temps, dec_temp_cont,\
        rews, rewards, conf]

lst = []
for i in product(*arrays):
    lst.append(list(i))

fig = plot_all(lst, hs=hs,utility=utility,testing=False)

