


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:09:11 2021
@author: sarah
"""


from curses import erasechar
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


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap






def run_single_inference(l):

    """load data"""

    switch, degr,learn_rew, q, p, h, tb, db, tpb, dec_temp,  rew, utility, config, infer_h, infer_dec, task, wi = l
    infer_both = infer_h and infer_dec

    prefix = task
    run_name = prefix+"hier_switch"+str(int(switch)) +"_degr"+str(int(degr)) +"_p"+str(p)+ "_learn_rew"+str(int(learn_rew))+\
            "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) + '_dec' + str(dec_temp) + \
            '_rew' + str(rew) + '_u' + '-'.join([str(u) for u in utility]) + '_' + config + '_rep' + str(wi) + "_extinguish.json"


    inference_file = 'inferences/inf_' + str(int(infer_h)) + str(int(infer_dec)) + '_' + run_name[:-5] + '_rep' + str(wi) + '.json'

    if not os.path.exists(os.path.join(os.getcwd(), inference_file)):
            
        fname = os.path.join(os.path.join(os.getcwd(), 'data_for_fitting'), run_name)
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
        data["planets"] = ar.tensor(world.environment.planet_conf)
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
        ns = 6 #number of states
        no = 2 #number of observations
        na = 2 #number of actionu
        npi = na**(T-1)
        nr = 3
        npl = 3

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

        if learn_rew:
            C_alphas = ar.ones([nr, npl, nc])
        else:
            C_alphas = ar.as_tensor(np.tile(planet_reward_probs.T[:,:,None]*5,(1,1,nc))+1)
            
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


        if infer_dec:
            dec_temp = array([1])
        else:
            dec_temp = array([dec_temp])

        # perception
        bayes_prc = prc.FittingPerception(
            A, 
            B, 
            C_agent, 
            transition_matrix_context, 
            state_prior, 
            utility, 
            prior_pi, 
            pol,
            pol_par,
            C_alphas,
            generative_model_context=C,
            T=T, 
            trials=trials,npart=n_part,dec_temp=dec_temp, r_lambda=rew)

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
                        number_of_rewards = nr, npart=n_part)
        lr = .02
        param_file = os.path.join('data_for_fitting/', run_name[:-5]   + '_' + str(int(infer_h)) + str(int(infer_dec))+'_'+ str(lr)+ '.save')

        ###################################
        """run inference"""

        params_dict = {'infer_h':infer_h, 'infer_dec': infer_dec, 'infer_both': infer_both}
        print('\n\ninference for:')
        print(run_name)
        print('\ninferring:')
        print(params_dict)
        print('agent vals pre inference ', 'dec: ', bayes_prc.dec_temp, ' h: ', bayes_prc.alpha_0)

        inferrer = inf.SingleInference(agent, data, params_dict)

        if os.path.exists(os.path.join(os.getcwd(), param_file)):
            inferrer.init_svi(num_particles=n_part,optim_kwargs={'lr':lr})
            inferrer.load_parameters(param_file)

            num_steps = 700
        else:
            num_steps = 700
        
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
            storage_name = os.path.join('data_for_fitting/', run_name[:-5]   + '_' + str(int(infer_h)) + str(int(infer_dec))+'_'+ str(lr)+ '.save')
            inferrer.save_parameters(storage_name)
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
                inferrer.load_parameters(storage_name)

        inferred_params['infer_h'] = infer_h
        inferred_params['infer_dec'] = infer_h
        inferred_params['n_part'] = n_part
        inferred_params['steps'] = its
        inferred_params['loss'] = loss.tolist()
        inferred_params['converged'] = converged
        inferred_params['tol'] = inferrer.tol
        inferred_params['rep'] = wi
        inferred_params['lr'] = lr
        name_root = 'figs/' + run_name[:-5] + '_rep' + str(wi) + '_' + str(int(infer_h)) + str(int(infer_dec))
        print('\n\ninference for:')
        print(run_name)
        print('\ninferring:')
        print(params_dict)
        inferrer.plot_posteriors(run_name = name_root)
        fig = plt.figure()
        plt.title("ELBO")
        plt.plot(np.arange((i)*size_chunk), inferrer.loss)
        plt.ylabel("ELBO")
        plt.xlabel("iteration")
        fig.savefig(name_root + '.png', dpi=300)
        
        with open('inferences/inf_' + str(int(infer_h)) + str(int(infer_dec)) + '_' + run_name[:-5] + '_rep' + str(wi) + '.json', 'w') as outfile:
            json.dump(inferred_params, outfile)

    else:
        print('ALREADY INFERRED')
        pass

def pooled(files_to_fit):
    # 17
    with Pool(20) as pool:

        for _ in tqdm.tqdm(pool.map(run_single_inference, files_to_fit),\
                        total=len(files_to_fit)):
            pass






data_folder = 'data_for_fitting'
path = os.path.join(os.getcwd(), data_folder)

if not os.path.exists(path):
    os.makedirs(path)
existing_files = os.listdir(path)



#############################################################################

# h =  [1,2,3,4,5,6,10,30,60,90,100]
# # h =  [2]
# h =[50,10,1]
# # h = [40]
# cue_ambiguity = [0.8]                       
# context_trans_prob = [0.85]
# cue_switch = [False]
# reward_naive = [True]
# training_blocks = [2]
# degradation_blocks=[1]
# degradation = [True]
# trials_per_block=[70]
# dec_temps = [1,2,4]
# rews = [0]
# # utility = [[1, 9 , 90], [5,25,70],[1,1,98],[1, 19, 80]]
# utility = [[1, 9, 90]]




#############################################################################

h =  [100,10,1]
# h = [40]
cue_ambiguity = [0.85]                       
context_trans_prob = [0.95]
cue_switch = [False]
reward_naive = [True]
training_blocks = [3]
degradation_blocks=[1]
degradation = [True]
trials_per_block=[70]
dec_temps = [2,4]
rews = [0]
utility = [[1,9,90]]
conf = ['original']
infer_h = [True]
infer_dec = [True]
task = ['']

arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
          training_blocks, degradation_blocks, trials_per_block,dec_temps,rews, utility,\
          conf, infer_h, infer_dec, task]


# n_part = 1
# lst = []
# for i in product(*arrays):
#     lst.append(list(i))

# folder = 'temp/'
# repetitions = 10
# files_to_fit = []

# for l in lst:

#     switch, degr,learn_rew, q, p, h, tb, db, tpb, dec_temp,  rew, utility, config, infer_h, infer_dec, task = l
#     infer_both = infer_h and infer_dec

#     prefix = task
#     name = prefix+"hier_switch"+str(int(switch)) +"_degr"+str(int(degr)) +"_p"+str(p)+ "_learn_rew"+str(int(learn_rew))+\
#             "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) + '_dec' + str(dec_temp) + \
#             '_rew' + str(rew) + '_u' + '-'.join([str(u) for u in utility]) + '_' + config
    
#     fit_run_name = name + '_rep' + str(repetitions) +  "_extinguish.json"

#     if not fit_run_name in existing_files:
#         fname = os.path.join(os.getcwd(), os.path.join(folder + config, name +  "_extinguish.json"))
#         jsonpickle_numpy.register_handlers()
        
#         with open(fname, 'r') as infile:
#             loaded = json.load(infile)

#         worlds = pickle.decode(loaded)
#         meta = worlds[-1]

#         for wi, world in enumerate(worlds[:-1]):
#             individual_world = [world,meta]
#             fname = os.path.join(os.getcwd(), data_folder + '/' + name  + '_rep' + str(wi) +  "_extinguish.json")
#             files_to_fit.append(l + [wi])
#             jsonpickle_numpy.register_handlers()
#             pickled = pickle.encode(individual_world)
#             with open(fname, 'w') as outfile:
#                 json.dump(pickled, outfile)
#     else:
#         for rep in range(repetitions):
#             fname = os.path.join(os.getcwd(), data_folder + '/' + name  + '_rep' + str(rep) +  "_extinguish.json")
#             files_to_fit.append(l + [rep])


# pooled(files_to_fit)


h =  [100,10,1]
# h = [40]
cue_ambiguity = [0.8]                       
context_trans_prob = [0.85]
cue_switch = [False]
reward_naive = [True]
training_blocks = [3]
degradation_blocks=[1]
degradation = [True]
trials_per_block=[70]
dec_temps = [2,4]
rews = [0]

utility = [[1, 9 , 90]]


conf = ['shuffled']
infer_h = [True]
infer_dec = [True]
task = ['multiple_']

arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
          training_blocks, degradation_blocks, trials_per_block,dec_temps,rews, utility,\
          conf, infer_h, infer_dec, task]


n_part = 1
lst = []
for i in product(*arrays):
    lst.append(list(i))

folder = 'temp/'
repetitions = 1
files_to_fit = []

for l in lst:

    switch, degr,learn_rew, q, p, h, tb, db, tpb, dec_temp,  rew, utility, config, infer_h, infer_dec, task = l
    infer_both = infer_h and infer_dec

    prefix = task
    name = prefix+"hier_switch"+str(int(switch)) +"_degr"+str(int(degr)) +"_p"+str(p)+ "_learn_rew"+str(int(learn_rew))+\
            "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) + '_dec' + str(dec_temp) + \
            '_rew' + str(rew) + '_u' + '-'.join([str(u) for u in utility]) + '_' + config
    
    fit_run_name = name + '_rep' + str(repetitions) +  "_extinguish.json"

    if not fit_run_name in existing_files:
        fname = os.path.join(os.getcwd(), os.path.join(folder + config, name +  "_extinguish.json"))
        jsonpickle_numpy.register_handlers()
        
        with open(fname, 'r') as infile:
            loaded = json.load(infile)

        worlds = pickle.decode(loaded)
        meta = worlds[-1]

        for wi, world in enumerate(worlds[:-1]):
            individual_world = [world,meta]
            fname = os.path.join(os.getcwd(), data_folder + '/' + name  + '_rep' + str(wi) +  "_extinguish.json")
            files_to_fit.append(l + [wi])
            jsonpickle_numpy.register_handlers()
            pickled = pickle.encode(individual_world)
            with open(fname, 'w') as outfile:
                json.dump(pickled, outfile)
    else:
        for rep in range(repetitions):
            fname = os.path.join(os.getcwd(), data_folder + '/' + name  + '_rep' + str(rep) +  "_extinguish.json")
            files_to_fit.append(l + [rep])


pooled(files_to_fit)

#%%%

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Sep 13 14:09:11 2021
# @author: sarah
# """


# import torch as ar
# array = ar.tensor

# import pyro
# import pyro.distributions as dist
# import agent as agt
# import perception as prc
# import action_selection as asl
# import inference_habit as inf
# import action_selection as asl
# import numpy as np
# import itertools
# import matplotlib.pylab as plt
# from matplotlib.animation import FuncAnimation
# from multiprocessing import Pool
# from matplotlib.colors import LinearSegmentedColormap
# import jsonpickle as pickle
# import jsonpickle.ext.numpy as jsonpickle_numpy
# import json
# import seaborn as sns
# import pandas as pd
# import os
# import scipy as sc
# import scipy.signal as ss
# import gc
# from environment import PlanetWorld
# from world import World
# import jsonpickle.ext.numpy as jsonpickle_numpy
# from itertools import product, repeat
# import multiprocessing.pool as mpp
# import tqdm


# def istarmap(self, func, iterable, chunksize=1):
#     """starmap-version of imap
#     """
#     self._check_running()
#     if chunksize < 1:
#         raise ValueError(
#             "Chunksize must be 1+, not {0:n}".format(
#                 chunksize))

#     task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
#     result = mpp.IMapIterator(self)
#     self._taskqueue.put(
#         (
#             self._guarded_task_generation(result._job,
#                                           mpp.starmapstar,
#                                           task_batches),
#             result._set_length
#         ))
#     return (item for chunk in result for item in chunk)


# mpp.Pool.istarmap = istarmap






# def run_single_inference(l):

#     """load data"""

#     switch, degr,learn_rew, q, p, h, tb, db, tpb, dec_temp,  rew, utility, config, infer_h, infer_dec, task = l
#     infer_both = infer_h and infer_dec

#     prefix = task
#     run_name = prefix+"hier_switch"+str(int(switch)) +"_degr"+str(int(degr)) +"_p"+str(p)+ "_learn_rew"+str(int(learn_rew))+\
#             "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) + '_dec' + str(dec_temp) + \
#             '_rew' + str(rew) + '_u' + '-'.join([str(u) for u in utility]) + '_' + config + "_extinguish.json"

#     fname = os.path.join(folder + config , run_name)
#     jsonpickle_numpy.register_handlers()

#     with open(fname, 'r') as infile:
#         loaded = json.load(infile)

#     worlds = pickle.decode(loaded)
#     meta = worlds[-1]
    
#     for wi, world in enumerate(worlds[:-1]):
#         data = {}
#         data["actions"] = ar.tensor(world.actions)
#         data["rewards"] = ar.tensor(world.rewards)
#         data["observations"] = ar.tensor(world.observations)    
#         data["context_obs"] = ar.tensor(world.environment.context_cues)
#         data["planets"] = ar.tensor(world.environment.planet_conf)
#         # data["actions"] = world.actions
#         # data["rewards"] = world.rewards
#         # data["observations"] = world.observations
#         # data["context_obs"] = world.environment.context_cues
#         # data["planets"] = world.environment.planet_conf

#         # print(data['planets'][:10,:])
#         ###################################
#         """experiment parameters"""

#         trials =  world.trials #number of trials
#         T = world.T #number of time steps in each trial
#         ns = 6 #number of states
#         no = 2 #number of observations
#         na = 2 #number of actionu
#         npi = na**(T-1)
#         nr = 3
#         npl = 3

#         if infer_h:
#             learn_pol = 1
#         else:
#             learn_pol = h
            
#         learn_habit = True

#         utility = ar.tensor(world.agent.perception.prior_rewards)
#         """
#         create matrices
#         """

#         #generating probability of observations in each state
#         A = ar.eye(ns)


#         #state transition generative probability (matrix)
#         nc = 4
#         B = ar.zeros([ns,ns,na])

#         m = [1,2,3,4,5,0]
#         for r, row in enumerate(B[:,:,0]):
#             row[m[r]] = 1

#         j = array([5,4,5,6,2,2])-1
#         for r, row in enumerate(B[:,:,1]):
#             row[j[r]] = 1

#         B = np.transpose(B, axes=(1,0,2))
#         B = np.repeat(B[:,:,:,None], repeats=nc, axis=3)



#         planet_reward_probs = array([[0.95, 0   , 0   ],
#                                         [0.05, 0.95, 0.05],
#                                         [0,    0.05, 0.95]]).T    # npl x nr

#         planet_reward_probs_switched = array([[0   , 0    , 0.95],
#                                                 [0.05, 0.95 , 0.05],
#                                                 [0.95, 0.05 , 0.0]]).T 


#         Rho = ar.zeros([trials, nr, ns])
#         planets = array(world.environment.planet_conf)
#         for i, pl in enumerate(planets):
#             if i >= tpb*tb and i < tpb*(tb + db) and degr == 1:
#                 probs = planet_reward_probs_switched
#             else:
#                 probs = planet_reward_probs

                
#             try:
#                 Rho[i,:,:] = probs[tuple([pl])].T
#             except:
#                 Rho[i,:,:] = probs[tuple([pl.long()])].T
#     # Rho[i,:,:] = planet_reward_probs[tuple([pl])].T

#         Rho = ar.as_tensor(Rho)

#         if learn_rew:
#             C_alphas = ar.ones([nr, npl, nc])
#         else:
#             C_alphas = ar.as_tensor(np.tile(planet_reward_probs.T[:,:,None]*5,(1,1,nc))+1)
            
#         C_agent = ar.zeros((nr, npl, nc))
#         for c in range(nc):
#             for pl in range(npl):
#                 C_agent[:,pl,c] = C_alphas[:,pl,c]/ C_alphas[:,pl,c].sum() 
#         #         array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(npl)]).T


#         r = (1-q)/(nc-1)

#         transition_matrix_context = np.zeros([nc,nc]) + r
#         transition_matrix_context = ar.as_tensor(transition_matrix_context - np.eye(nc)*r + np.eye(nc)*q)

#         """
#         create policies
#         """

#         pol = array(list(itertools.product(list(range(na)), repeat=T-1)))

#         npi = pol.shape[0]

#         # prior over policies
#         # prior_pi = ar.ones(npi)/npi #ar.zeros(npi) + 1e-3/(npi-1)
#         alphas = ar.zeros((npi,nc)) + learn_pol
#         prior_pi = alphas / alphas[:,0].sum()
#         alphas = array([learn_pol])

#         """
#         set state prior (where agent thinks it starts)
#         """

#         state_prior = ar.ones((ns))
#         state_prior = state_prior/ state_prior.sum()

#         prior_context = array([0.99] + [(1-0.99)/(nc-1)]*(nc-1))


#         no =  2                                        # number of observations (background color)
#         C = ar.zeros([no, nc])
#         dp = 0.001
#         p2 = 1 - p - dp
#         C[0,:] = array([p,dp/2,p2,dp/2])
#         C[1,:] = array([dp/2, p, dp/2, p2])

#         """
#         set up agent
#         """
#         #bethe agent

#         pol_par = alphas


#         if infer_dec:
#             dec_temp = array([1])
#         else:
#             dec_temp = array([dec_temp])

#         # perception
#         bayes_prc = prc.FittingPerception(
#             A, 
#             B, 
#             C_agent, 
#             transition_matrix_context, 
#             state_prior, 
#             utility, 
#             prior_pi, 
#             pol,
#             pol_par,
#             C_alphas,
#             generative_model_context=C,
#             T=T, 
#             trials=trials,npart=n_part,dec_temp=dec_temp, r_lambda=rew)

#         ac_sel = asl.AveragedSelector(trials = trials, T = T,
#                                             number_of_actions = na)

#         agent = agt.FittingAgent(bayes_prc, ac_sel, pol,
#                         trials = trials, T = T,
#                         prior_states = state_prior,
#                         prior_policies = prior_pi,
#                         number_of_states = ns,
#                         prior_context = prior_context,
#                         learn_habit = learn_habit,
#                         learn_rew = True,
#                         #save_everything = True,
#                         number_of_policies = npi,
#                         number_of_rewards = nr, npart=n_part)



#         ###################################
#         """run inference"""

#         params_dict = {'infer_h':infer_h, 'infer_dec': infer_dec, 'infer_both': infer_both}
#         print('\n\ninference for:')
#         print(run_name)
#         print('\ninferring:')
#         print(params_dict)
#         print('agent vals pre inference ', 'dec: ', bayes_prc.dec_temp, ' h: ', bayes_prc.alpha_0)

#         inferrer = inf.SingleInference(agent, data, params_dict)
#         # loss = inferrer.infer_posterior(iter_steps=iss, num_particles=n_part)


        # num_steps = 500
        # size_chunk = 100
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


#         num_steps = 500
#         size_chunk = 100
#         converged = False
#         max_steps = False
#         i = 0
#         while not converged and not max_steps:
#         # for i in range(num_steps//size_chunk):
#             loss = inferrer.infer_posterior(iter_steps=size_chunk, num_particles=n_part)#, param_dict
#             total_num_iter_so_far = (i+1)*size_chunk
#             storage_name = os.path.join(folder, run_name[:-5]   + '_' + str(int(infer_h)) + str(int(infer_dec))+'.save')
#             inferrer.save_parameters(storage_name)
#             converged = inferrer.check_convergence(loss)
#             i += 1
#             its = i*size_chunk
#             if converged:
#                 inferred_params = inferrer.return_inferred_parameters()
#                 converged = True
#             elif (its >= num_steps):
#                 inferred_params = inferrer.return_inferred_parameters()
#                 max_steps = True
#             else:
#                 inferrer.load_parameters(storage_name)


#         inferred_params['infer_h'] = infer_h
#         inferred_params['infer_dec'] = infer_h
#         inferred_params['n_part'] = n_part
#         inferred_params['steps'] = its
#         inferred_params['loss'] = loss.tolist()
#         inferred_params['converged'] = converged
#         inferred_params['tol'] = inferrer.tol
#         name_root = 'figs/' + run_name[:-5] + '_rep' + str(wi) + '_' + str(int(infer_h)) + str(int(infer_dec))
#         print('\n\ninference for:')
#         print(run_name)
#         print('\ninferring:')
#         print(params_dict)
#         inferrer.plot_posteriors(run_name = name_root)
#         fig = plt.figure()
#         plt.title("ELBO")
#         plt.plot(np.arange((i)*size_chunk), inferrer.loss)
#         plt.ylabel("ELBO")
#         plt.xlabel("iteration")
#         fig.savefig(name_root + '.png', dpi=300)
        
#         with open('inferences/inf_' + str(int(infer_h)) + str(int(infer_dec)) + '_' + run_name[:-5] + '_rep' + str(wi) + '.json', 'w') as outfile:
#             json.dump(inferred_params, outfile)
#         # plt.show()

# def pooled(lst):

#     with Pool(11) as pool:

#         for _ in tqdm.tqdm(pool.map(run_single_inference, lst),\
#                         total=len(lst)):
#             pass



# h =  [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
# # h = [40]
# cue_ambiguity = [0.75]                       
# context_trans_prob = [0.85]
# cue_switch = [False]
# reward_naive = [True]
# training_blocks = [4]
# degradation_blocks=[2]
# degradation = [True]
# trials_per_block=[70]
# dec_temps = [1]
# rews = [0]
# utility = [[1,19 , 0]]
# conf = ['shuffled']
# task = ['multiple_']
# infer_h = [True]
# infer_dec = [True]

# arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
#           training_blocks, degradation_blocks, trials_per_block,dec_temps,rews, utility,\
#           conf, infer_h, infer_dec, task]
# n_part = 30


# folder = 'temp/'

# lst = []
# for i in product(*arrays):
#     lst.append(list(i))

# pooled(lst)