


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



def run_single_inference(fname):

    if sys.platform == 'win32':
        splitter = '\\'
    else:
        splitter = '/'
    inference_file = 'inferences' + splitter + 'inf_' + fname.split(splitter)[-1]

    if not os.path.exists(os.path.join(os.getcwd(), inference_file)):
            
        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            loaded = json.load(infile)

        worlds = pickle.decode(loaded)
        world = worlds[0]
        meta = worlds[-1]
        
        data = {}
        data["actions"] = ar.tensor(world.actions)
        data["rewards"] = ar.tensor(world.rewards)
        data["observations"] = ar.tensor(world.observations)    
        data["context_obs"] = ar.tensor(world.environment.context_cues)
        data["planets"] = ar.tensor(world.environment.planet_conf, dtype=ar.long)

        # data["actions"] = world.actions
        # data["rewards"] = world.rewards
        # data["observations"] = world.observations
        # data["context_obs"] = world.environment.context_cues
        # data["planets"] = world.environment.planet_conf

        # print(data['planets'][:10,:])
        ###################################
        """experiment parameters"""

        trials =  world.trials #number of trials
        T = world.T #number of time steps in each trial
        npi = na**(T-1)
        # ns = 6 #number of states
        # no = 2 #number of observations
        # na = 2 #number of actionu
        # nr = 3
        # npl = 3

        if infer_h:
            learn_pol = 1
        else:
            learn_pol = h
            
        learn_habit = True

        utility = ar.tensor(world.agent.perception.prior_rewards)
        """
        create matrices
        """

        #generating probability of observations in each state
        A =  array(world.agent.perception.generative_model_observations)
        B =  array(world.agent.perception.generative_model_states)
        Rho = array(world.environment.R)

        if learn_rew:
            C_beta = ar.ones([nr, npl, nc])
        else:
            # C_beta = ar.as_tensor(np.tile(planet_reward_probs.T[:,:,None]*5,(1,1,nc))+1)
            C_beta = array(world.agent.perception.dirichlet_rew_params_init)

        C_agent = ar.zeros((nr, npl, nc))
        for c in range(nc):
            for pl in range(npl):
                C_agent[:,pl,c] = C_beta[:,pl,c]/ C_beta[:,pl,c].sum() 


        transition_matrix_context =array(world.agent.perception.transition_matrix_context)


        pol = array(world.agent.perception.policies)

        npi = pol.shape[0]

        # prior over policies
        # prior_pi = ar.ones(npi)/npi #ar.zeros(npi) + 1e-3/(npi-1)

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
        #bethe agent

        alphas
        # dec_temp_cont = array(world.agent.perception.dec_temp_cont)

        if infer_dec:
            dec_temp = array([1])
        else:
            dec_temp = array([dec_temp])

        # perception
        bayes_prc = prc.FittingPerception(
            A, 
            B, 
            C_agent, 
            state_prior, 
            utility, 
            prior_pi, 
            pol,
            alphas,
            C_beta,
            transition_matrix_context = transition_matrix_context, 
            generative_model_context=C, prior_context=prior_context, number_of_planets = npl,
            T=T, trials=trials, dec_temp=dec_temp, dec_temp_cont=dec_temp_cont, r_lambda=rew, npart=n_part)

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                            number_of_actions = na)

        agent = agt.FittingAgent(bayes_prc,
                        ac_sel,
                        pol,
                        prior_states = state_prior,
                        prior_policies = prior_pi,
                        prior_context = prior_context, number_of_planets=npl,
                        number_of_policies = npi, number_of_rewards = nr, trials = trials, T=T,
                        number_of_states = ns,
                        )
        lr = .01
        param_file = inference_file[:-5]   + '_' + str(int(infer_h)) + str(int(infer_dec))+'_'+ str(lr)+ '.save'
        if sys.platform == 'win32':
           param_file =  param_file.replace('/','\\')
        ###################################
        """run inference"""

        params_dict = {'infer_h':infer_h, 'infer_dec': infer_dec, 'infer_both': infer_both}
        print('\n\ninference for:')
        print('\ninferring:')
        print(params_dict)
        print('agent vals pre inference ', 'dec: ', bayes_prc.dec_temp, ' h: ', bayes_prc.alpha_0)

        inferrer = inf.SingleInference(agent, data, params_dict)

        if os.path.exists(os.path.join(os.getcwd(), param_file)):
            inferrer.init_svi(num_particles=n_part,optim_kwargs={'lr':lr})
            inferrer.load_parameters(param_file)
            
        num_steps = 2500
        size_chunk = 50
        converged = False
        max_steps = False
        i = 0


        while not converged and not max_steps:  
        # for i in range(num_steps//size_chunk):
            # loss = inferrer.infer_posterior(iter_steps=size_chunk, num_particles=n_part,optim_kwargs={'lr':0.05})#, param_dict
            loss = inferrer.infer_posterior(iter_steps=size_chunk, num_particles=n_part,optim_kwargs={'lr':lr})#,total_num_iter_so_far = (i+1)*size_chunk
            total_num_iter_so_far = (i+1)*size_chunk
            print('total steps:', total_num_iter_so_far)
            inferrer.save_parameters(param_file)
            converged = inferrer.check_convergence(loss)
            i += 1
            its = i*size_chunk
            if converged:
                inferred_params = inferrer.return_inferred_parameters()
                converged = True
            elif (its >= num_steps):
                inferred_params = inferrer.return_inferred_parameters()
                max_steps = True
            else:
                inferrer.load_parameters(param_file)

        inferred_params['infer_h'] = infer_h
        inferred_params['infer_dec'] = infer_h
        inferred_params['n_part'] = n_part
        inferred_params['steps'] = its
        inferred_params['loss'] = loss.tolist()
        inferred_params['converged'] = converged
        inferred_params['tol'] = inferrer.tol
        inferred_params['rep'] = wi
        inferred_params['lr'] = lr
        name_root = 'figs' + splitter + '_'.join(inference_file.split(splitter)[-1].split('_')[:-1])
        print('\n\ninference for:')
        print(inference_file)
        print('\ninferring:')
        print(params_dict)
        inferrer.plot_posteriors(run_name = name_root)
        fig = plt.figure()
        plt.title("ELBO")
        plt.plot(np.arange((i)*size_chunk), inferrer.loss)
        plt.ylabel("ELBO")
        plt.xlabel("iteration")
        fig.savefig(name_root + '.png', dpi=300)
        
        with open(inference_file, 'w') as outfile:
            json.dump(inferred_params, outfile)

    else:
        print('ALREADY INFERRED')
        pass

def pooled(files_to_fit):
    # 17; 20
    
    if __name__ == '__main__':
        with Pool(20) as pool:

            for _ in tqdm.tqdm(pool.map(run_single_inference, files_to_fit),\
                            total=len(files_to_fit)):
                pass
        # run_single_inference(files_to_fit[0])

task = ['multiple_']

arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
          training_blocks, degradation_blocks, trials_per_block,dec_temps, dec_context, rews, utility, \
          conf, task]

lst = []
for i in product(*arrays):
    lst.append(list(i))


n_part = 1
repetitions = 1
files_to_fit = []
data_folder = 'temp'

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


if __name__ == '__main__':
    # run_single_inference(files_to_fit[0])
    pooled(files_to_fit)
