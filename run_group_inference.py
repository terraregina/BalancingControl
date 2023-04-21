


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021
@author: sarah
"""


# from curses import erasechar
import torch as ar
array = ar.tensor

import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl
import inference_habit as inf
import action_selection as asl
import numpy as np
import itertools
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import pandas as pd
import os
import scipy as sc
import scipy.signal as ss
import gc
from environment import PlanetWorld
from world import World
import jsonpickle.ext.numpy as jsonpickle_numpy
from itertools import product, repeat
import multiprocessing.pool as mpp
import tqdm
from sim_parameters import *
import sys
# from misc import istarmap

def sample_posterior(inferrer, prefix, total_num_iter_so_far, n_samples=500,data=None):

    sample_df = inferrer.sample_posterior(n_samples=n_samples) #inferrer.plot_posteriors(n_samples=1000)
    sample_df[['inferred_h', 'inferred_dec_temp']] = sample_df.groupby('subject').transform('mean')
    # sample_df.sort_values('subject', inplace=True)

    true_dt = [val['dec_temp'] for val in true_vals]
    true_h = [val['h'] for val in true_vals]


    sample_df = sample_df.copy()
    sample_df['true_dec_temp'] = ar.tensor(true_dt).repeat(n_samples)
    sample_df['true_h'] = ar.tensor(true_h).repeat(n_samples)

    sample_file = prefix+'recovered_samples_'+str(total_num_iter_so_far)+'_'+str(sample_df.subject.unique().size)+'agents.csv'
    fname = os.path.join(os.getcwd()+ "/" + "inferences" + "/" , sample_file)
    if sys.platform == 'win32':
        fname = fname.replace('/','\\')
    sample_df.to_csv(fname)

    return sample_df


def plot_posterior(total_df, total_num_iter_so_far, prefix):

    if sys.platform == 'win32':
        splitter = '\\'
    else:
        splitter = '/'
    
    plt.figure()
    sns.scatterplot(data=total_df, x="true_dec_temp", y="inferred_dec_temp")
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.xlabel("true dec_temp")
    plt.ylabel("inferred dec_temp")
    plt.savefig(os.getcwd() + splitter + 'inferences' + splitter + prefix+"recovered_"+str(total_num_iter_so_far)+"_"+str(total_df.subject.unique().size)+"agents_dec_temp.png")
    plt.close()
    # plt.show()

    plt.figure()
    sns.scatterplot(data=total_df, x="true_h", y="inferred_h")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    # plt.ylim([-0.1, 110])
    plt.xlabel("true h")
    plt.ylabel("inferred h")
    plt.savefig(os.getcwd() + splitter + 'inferences' + splitter + prefix+"recovered_"+str(total_num_iter_so_far)+"_"+str(total_df.subject.unique().size)+"agents_h.png")
    # plt.show()
    plt.close()

def load_structured_data(fname):
    data = {}
    data["actions"] = []
    data["rewards"] = []
    data["observations"] = []
    data["context_obs"] = []
    data["planets"] = []

    for f in fname:
        jsonpickle_numpy.register_handlers()

        with open(f, 'r') as infile:
            loaded = json.load(infile)

        worlds = pickle.decode(loaded)
        world = worlds[0]
        meta = worlds[-1]

        data["actions"].append(ar.tensor(world.actions))
        data["rewards"].append(ar.tensor(world.rewards))
        data["observations"].append(ar.tensor(world.observations))    
        data["context_obs"].append(ar.tensor(world.environment.context_cues,dtype=ar.long))
        data["planets"].append(ar.tensor(world.environment.planet_conf))

    for key in data.keys():
        data[key] = ar.stack(data[key],dim=-1)
    
    data['context_obs'] = data['context_obs'][...,0]
    data['planets'] = data['planets'][...,0]
    
    return data


def run_inference(fnames):

    if sys.platform == 'win32':
        splitter = '\\'
    else:
        splitter = '/'
            
    inference_file = 'inferences' + splitter + 'group_inf_' + '-'.join([str(h) for h in hs]) + '_reps' + str(repetitions)
    
    nsubs = len(fnames)
    print('subjects: ',nsubs)
    data = load_structured_data(fnames)
    with open(fnames[0], 'r') as infile:
        loaded = json.load(infile)

    worlds = pickle.decode(loaded)
    world = worlds[0]
    meta = worlds[-1]

    """experiment parameters"""

    trials =  world.trials #number of trials
    T = world.T #number of time steps in each trial
    npi = na**(T-1)

    if infer_h:
        learn_pol = 1
    else:
        learn_pol = h
        
    utility = ar.tensor(world.agent.perception.prior_rewards)
    """
    create matrices
    """

    #generating probability of observations in each state
    A =  array(world.agent.perception.generative_model_observations)
    B =  array(world.agent.perception.generative_model_states)
    Rho = array(world.environment.R)

    if learn_rew:
        C_betas = ar.ones([nr, npl, nc])
    else:
        # C_betas = ar.as_tensor(np.tile(planet_reward_probs.T[:,:,None]*5,(1,1,nc))+1)
        C_betas = array(world.agent.perception.dirichlet_rew_params_init)

    C_agent = ar.zeros((nr, npl, nc))
    for c in range(nc):
        for pl in range(npl):
            C_agent[:,pl,c] = C_betas[:,pl,c]/ C_betas[:,pl,c].sum() 


    transition_matrix_context =array(world.agent.perception.transition_matrix_context)


    pol = array(world.agent.perception.policies)

    npi = pol.shape[0]

    alphas = ar.zeros((npi,nc)) + learn_pol
    prior_pi = alphas / alphas[:,0].sum()
    alphas = array([learn_pol])

    """
    set state prior (where agent thinks it starts)
    """

    state_prior = ar.ones((ns))
    state_prior = state_prior/ state_prior.sum()
    
    # not directly fed by the agent
    prior_context = ar.zeros((nc)) + 0.1/(nc-2)
    prior_context[:2] = (1 - 0.1)/2

    C = array(world.agent.perception.generative_model_context)

    """
    set up agent
    """

    pol_par = alphas

    if infer_dec:
        dec_temp = array([1])
    else:
        dec_temp = array([dec_temp])

    # perception
    bayes_prc = prc.GroupFittingPerception(
        A, 
        B, 
        C_agent, 
        state_prior, 
        utility, 
        prior_pi, 
        pol,
        pol_par,
        C_betas,
        transition_matrix_context = transition_matrix_context, 
        generative_model_context=C, prior_context=prior_context,
        number_of_planets = npl, 
        T=T, dec_temp = dec_temp, dec_temp_cont = dec_temp_cont,
        npart=n_part, nsubs = nsubs, nr=nr, trials=trials)
    
    bayes_prc.npars = 2
    bayes_prc.mask = None

    agent = agt.FittingAgent(bayes_prc, [], pol,
                    prior_states = state_prior,
                    prior_policies = prior_pi,
                    prior_context = prior_context, number_of_planets = npl,
                    number_of_policies = npi, number_of_rewards = nr, trials = trials, T = T,
                    number_of_states = ns)
    
    inferrer = inf.GeneralGroupInference(agent, data)
    param_file = inference_file + '_infh' + str(int(infer_h)) + '_infd' +  str(int(infer_dec)) +'_' + str(lr)+ '.save'
    if sys.platform == 'win32':
        param_file =  param_file.replace('/','\\')
    ###################################
    """run inference"""

    params_dict = {'infer_h':infer_h, 'infer_dec': infer_dec, 'infer_both': infer_both}
    print('\n\ninference for:')
    print('\ninferring:')
    print(params_dict)
    print('agent vals pre inference ', 'dec: ', bayes_prc.dec_temp, ' h: ', bayes_prc.alpha_0)


    if os.path.exists(os.path.join(os.getcwd(), param_file)):
        inferrer.init_svi(num_particles=n_part,optim_kwargs={'lr':lr})
        inferrer.load_parameters(param_file)
        
    num_steps = 2000
    size_chunk = 50
    # num_steps = 2
    # size_chunk = 1
    converged = False
    max_steps = False
    i = 0

    while not converged and not max_steps:  
    # for i in range(num_steps//size_chunk):
        # loss = inferrer.infer_posterior(iter_steps=size_chunk, num_particles=n_part,optim_kwargs={'lr':0.05})#, param_dict
        loss = inferrer.infer_posterior(iter_steps=size_chunk, num_particles=n_part,optim_kwargs={'lr':lr})#,total_num_iter_so_far = (i+1)*size_chunk
        its = i*size_chunk
        total_num_iter_so_far = (i+1)*size_chunk
        print('total steps:', total_num_iter_so_far)
        inferrer.save_parameters(param_file)
        # converged = inferrer.check_convergence(loss)
        converged = False
        inferrer.agent.perception.param_names = list(inferrer.agent.perception.locs_to_pars(ar.zeros(2)).keys())
        full_df = sample_posterior(inferrer, str(its) + '_'+prefix, 100, data=data)
        plot_posterior(full_df, 100, prefix)
        i += 1
        if converged:
            # inferred_params = inferrer.return_inferred_parameters()
            converged = True
            print('inference finished')
        elif (its >= num_steps):
            # inferred_params = inferrer.return_inferred_parameters()
            max_steps = True
            print('ínference finished')
        else:
            inferrer.load_parameters(param_file)
        

def start_inference(files_to_fit,pooled=False):
    # pooled artifact from before
    if pooled:
        with Pool(2) as pool:
            for _ in tqdm.tqdm(pool.map(run_inference, files_to_fit),\
                            total=len(files_to_fit)):
                    pass
    else:
        run_inference(files_to_fit)


''' Unpack simulation files and load true values '''

lr = .01
n_part = 10
repetitions = 1
data_folder = 'temp'

lst = []
files_to_fit = []
true_vals = []

task = ['multiple_']

arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
          training_blocks, degradation_blocks, trials_per_block,dec_temps, dec_context, rews, utility, \
          conf, task]


for i in product(*arrays):
    lst.append(list(i))


for l in lst:
    switch, degr,learn_rew, q, p, h, tb, db, tpb, dec_temp, dec_temp_cont, rew, utility, config, task = l
    infer_both = infer_h and infer_dec

    prefix = task

    if use_fitting:
        prefix += 'fitt_switch'
    else:
        prefix += 'hier_switch'

    name = prefix+str(int(switch)) +"_degr"+str(int(degr)) +"_p"+str(p)+ "_learn_rew"+str(int(learn_rew))+\
            "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) + '_decp' + str(dec_temp) + \
            '_decc' + str(dec_temp_cont) + '_rew' + str(rew) + '_u' + '-'.join([str(u) for u in utility]) + '_'+ str(nr) +  '_' + config
    
    fit_run_name = name + '_rep' + str(repetitions) +  ".json"

    fname = os.getcwd()  + '/' + data_folder + '/' + config + '/' + name +  ".json"
    if sys.platform == "win32":
        fname = fname.replace('/','\\')
    
    jsonpickle_numpy.register_handlers()
    with open(fname, 'r') as infile:
        loaded = json.load(infile)

    worlds = pickle.decode(loaded)
    meta = worlds[-1]

    for wi, world in enumerate(worlds[:-1]):
        individual_world = [world,meta]
        fname = os.path.join(os.getcwd(), data_folder + '/' + name  + '_rep' + str(wi) +  ".json")
        if sys.platform == "win32":
            fname = fname.replace('/','\\')

        # files_to_fit.append(l + [wi])
        files_to_fit.append(fname)
        jsonpickle_numpy.register_handlers()
        pickled = pickle.encode(individual_world)
        with open(fname, 'w') as outfile:
            json.dump(pickled, outfile)

        true_vals.append({"dec_temp": ar.tensor(dec_temp), "h": ar.tensor(1./h)})


if __name__ == '__main__':
    start_inference(files_to_fit,pooled=False)
    # start_inference(files_to_fit,pooled=True)