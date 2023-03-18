#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021
@author: sarah
"""


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
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from itertools import product
from sim_parameters import *
import sys

#ar.autograd.set_detect_anomaly(True)
###################################
ar.set_num_threads(1)

task =['multiple_']
folder = 'temp'
true_vals = []
data = []

# h =  [1, 2]#,3,4,5,6,7]#,8,9,10,20,30,40,50,60] #11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,200]
# cue_ambiguity = [0.9]#,0.8,0.95]                       
# context_trans_prob = [0.9]#, 0.95]                
# degradation = [True]
# cue_switch = [False]
# reward_naive = [False]
# training_blocks = [2]
# degradation_blocks=[2]
# trials_per_block=[70]
# dec_temps = [1]#,2,3,4,5,6]
# conf_folder = ['ordered']
# infer_h = [True]
# infer_dec = [True]
arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
          training_blocks, degradation_blocks, trials_per_block,dec_temps, dec_context, rews, utility, \
          conf, task]
lst = []
for i in product(*arrays):
    lst.append(list(i))


n_part = 5
for li, l in enumerate(lst):

    """load data"""

    switch, degr,learn_rew, q, p, h, tb, db, tpb, dec_temp, dec_temp_cont, rew, utility, config, task = l
    infer_both = infer_h and infer_dec

    prefix = task
    if use_fitting:
        prefix += 'fitt_switch'
    else:
        prefix += 'hier_switch'

    run_name = prefix+str(int(switch)) +"_degr"+str(int(degr)) +"_p"+str(p)+ "_learn_rew"+str(int(learn_rew))+\
            "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) + '_decp' + str(dec_temp) + \
            '_decc' + str(dec_temp_cont) + '_rew' + str(rew) + '_u' + '-'.join([str(u) for u in utility]) + '_'+ str(nr) +  '_' + config + '_extinguish.json'
    
    fname = os.getcwd() + '/' + folder + '/' + config +  '/' +  run_name

    if sys.platform == "win32":
        fname = fname.replace('/','\\')

    jsonpickle_numpy.register_handlers()

    with open(fname, 'r') as infile:
        loaded = json.load(infile)

    worlds = pickle.decode(loaded)
    # world = worlds[0]
    meta = worlds[-1]
    for world in worlds[:-1]:
        data.append({})
        data[-1]["actions"] = ar.tensor(world.actions)
        data[-1]["rewards"] = ar.tensor(world.rewards)
        data[-1]["observations"] = ar.tensor(world.observations)    
        data[-1]["context_obs"] = ar.tensor(world.environment.context_cues)
        data[-1]["planets"] = ar.tensor(world.environment.planet_conf)
        # data[-1]["planets"] = ar.tensor(world.environment.planet_conf)

        true_vals.append(meta)

nsubs = len(true_vals)
print('analyzing '+str(nsubs)+' data sets')

###################################
"""experiment parameters"""

trials =  world.trials #number of trials
T = world.T #number of time steps in each trial
ns = 6 #number of states
no = 2 #number of observations
na = 2 #number of actions
npi = na**(T-1)
nr = 3
npl = 3


learn_pol = h
# learn_rew = True
learn_habit = True

utility = []

ut = [0.999]
for u in ut:
    utility.append(ar.zeros(nr))
    for i in range(0,nr):
        utility[-1][i] = (1-u)/(nr-1)#u/nr*i
    utility[-1][-1] = u
    
utility = utility[-1]

"""
create matrices
"""

#generating probability of observations in each state
A = ar.eye(ns)


#state transition generative probability (matrix)
nc = 4
B = ar.zeros([ns,ns,na])

m = [1,2,3,4,5,0]
for r, row in enumerate(B[:,:,0]):
    row[m[r]] = 1

j = array([5,4,5,6,2,2])-1
for r, row in enumerate(B[:,:,1]):
    row[j[r]] = 1

B = np.transpose(B, axes=(1,0,2))
B = np.repeat(B[:,:,:,None], repeats=nc, axis=3)



planet_reward_probs = array([[0.95, 0   , 0   ],
                                [0.05, 0.95, 0.05],
                                [0,    0.05, 0.95]]).T    # npl x nr

planet_reward_probs_switched = array([[0   , 0    , 0.95],
                                        [0.05, 0.95 , 0.05],
                                        [0.95, 0.05 , 0.0]]).T 


Rho = ar.zeros([trials, nr, ns])
planets = array(world.environment.planet_conf)
for i, pl in enumerate(planets):
    if i >= tpb*tb and i < tpb*(tb + db) and degr == 1:
        probs = planet_reward_probs_switched
    else:
        probs = planet_reward_probs

        
    try:
        Rho[i,:,:] = probs[tuple([pl])].T
    except:
        Rho[i,:,:] = probs[tuple([pl.long()])].T
# Rho[i,:,:] = planet_reward_probs[tuple([pl])].T

Rho = ar.as_tensor(Rho)

# if learn_rew:
#     C_alphas = ar.ones([nr, npl, nc])
# else:
C_alphas = ar.as_tensor(np.tile(planet_reward_probs.T[:,:,None]*20,(1,1,nc))+1)
    
C_agent = ar.zeros((nr, npl, nc))
for c in range(nc):
    for pl in range(npl):
        C_agent[:,pl,c] = C_alphas[:,pl,c]/ C_alphas[:,pl,c].sum() 
#         array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(npl)]).T


r = (1-q)/(nc-1)

transition_matrix_context = np.zeros([nc,nc]) + r
transition_matrix_context = ar.as_tensor(transition_matrix_context - np.eye(nc)*r + np.eye(nc)*q)

"""
create policies
"""

pol = array(list(itertools.product(list(range(na)), repeat=T-1)))

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

prior_context = array([0.99] + [(1-0.99)/(nc-1)]*(nc-1))


no =  2                                        # number of observations (background color)
C = ar.zeros([no, nc])
dp = 0.001
p2 = 1 - p - dp
C[0,:] = array([p,dp/2,p2,dp/2])
C[1,:] = array([dp/2, p, dp/2, p2])

"""
set up agent
"""
#bethe agent

pol_par = alphas

data_obs = ar.stack([d["observations"] for d in data], dim=-1)
data_rew = ar.stack([d["rewards"] for d in data], dim=-1)
data_act = ar.stack([d["actions"] for d in data], dim=-1)
data_pln = data[-1]['planets']
data_cntxt = ar.stack([d['context_obs'] for d in data], dim=-1)
# data_cntxt = data[-1]['context_obs']
structured_data = {"observations": data_obs, "rewards": data_rew, "actions": data_act, "planets": data_pln, 'context': data_cntxt}


# if infer_dec:
#     dec_temp = array([1])
# else:
#     dec_temp = array([dec_temp])

# perception
bayes_prc = prc.GroupPerception(
                            A, 
                            B, 
                            C_agent, 
                            transition_matrix_context, 
                            state_prior, 
                            utility, 
                            prior_pi, 
                            pol,
                            alphas,
                            # pol_par,
                            C_alphas,
                            trials = trials,
                            pol_lambda = 0,
                            r_lambda = 0,
                            non_decaying = 0,
                            dec_temp=1,
                            generative_model_context=C,
                            T=T, 
                            npart=n_part, nsubs=nsubs)

ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                    number_of_actions = na)

agent = agt.FittingAgent(bayes_prc, ac_sel, pol,
                trials = trials, T = T,
                prior_states = state_prior,
                prior_policies = prior_pi,
                number_of_states = ns,
                prior_context = prior_context,
                learn_habit = learn_habit,
                learn_rew = True,
                #save_everything = True,
                number_of_policies = npi,
                number_of_rewards = nr, npart=n_part,nsubs=nsubs)



###################################
"""run inference"""




inferrer = inf.GroupInference(agent, structured_data)

num_steps = 500
size_chunk = num_steps

for i in range(num_steps//size_chunk):
    print('taking steps '+str(i*(size_chunk)+1)+' to '+str((i+1)*(size_chunk))+' out of total '+str(num_steps))
    inferrer.infer_posterior(iter_steps=size_chunk, num_particles=15)#, param_dict
    
    total_num_iter_so_far = i*size_chunk
    storage_name = 'recovered_'+str(total_num_iter_so_far)+'.save'#h_recovered
    storage_name = os.path.join(folder, storage_name)
    inferrer.save_parameters(storage_name)
    # inferrer.load_parameters(storage_name)
    
    loss = inferrer.loss
    plt.figure()
    plt.title("ELBO")
    plt.plot(loss)
    plt.ylabel("ELBO")
    plt.xlabel("iteration")
    plt.savefig('recovered_ELBO')
    plt.show()



# params_dict = {'infer_h':infer_h, 'infer_dec': infer_dec, 'infer_both': infer_both}
# print('\n\ninference for:')
# print(run_name)
# print('\ninferring:')
# print(params_dict)
# print('agent vals pre inference ', 'dec: ', bayes_prc.dec_temp, ' h: ', bayes_prc.alpha_0)

# inferrer = inf.SingleInference(agent, data, params_dict)
# loss = inferrer.infer_posterior(iter_steps=iss, num_particles=n_part)



# num_steps = 1000
# size_chunk = 50
# converged = False
# max_steps = False
# i = 0
# while not converged and not max_steps:
# # for i in range(num_steps//size_chunk):
#     loss = inferrer.infer_posterior(iter_steps=size_chunk, num_particles=n_part)#, param_dict
#     total_num_iter_so_far = (i+1)*size_chunk
#     storage_name = os.path.join(folder, run_name[:-5]   + '_' + str(int(infer_h)) + str(int(infer_dec))+'.save')
#     inferrer.save_parameters(storage_name)
#     converged = inferrer.check_convergence(loss)
#     i += 1
#     its = i*size_chunk
#     if converged:
#         inferred_params = inferrer.return_inferred_parameters()
#         converged = True
#     elif (its >= num_steps):
#         inferred_params = inferrer.return_inferred_parameters()
#         max_steps = True
#     else:
#         inferrer.load_parameters(storage_name)


# inferred_params['infer_h'] = infer_h
# inferred_params['infer_dec'] = infer_h
# inferred_params['n_part'] = n_part
# inferred_params['steps'] = its
# inferred_params['los                                          s'] = loss.tolist()
# inferred_params['converged'] = converged
# inferred_params['tol'] = inferrer.tol

# print('\n\ninference for:')
# print(run_name)
# print('\ninferring:')
# print(params_dict)
# inferrer.plot_posteriors(run_name = 'figs/' + run_name[:-5] + '_' + str(int(infer_h)) + str(int(infer_dec)))
# fig = plt.figure()
# plt.title("ELBO")
# plt.plot(np.arange((i)*size_chunk), inferrer.loss)
# plt.ylabel("ELBO")
# plt.xlabel("iteration")
# fig.savefig('figs/' + run_name[:-5] + '_' + str(int(infer_h)) + str(int(infer_dec)) + '.png', dpi=300)

# with open('inferences/inf_' + str(int(infer_h)) + str(int(infer_dec)) + '_' + run_name, 'w') as outfile:
#     json.dump(inferred_params, outfile)
# # plt.show()

