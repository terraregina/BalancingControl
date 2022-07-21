# # %%
# Imports
# 

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
                    repetitions, use_fitting):


    switch_cues, contingency_degradation, reward_naive, context_trans_prob, cue_ambiguity, h,\
    training_blocks, degradation_blocks, trials_per_block, dec_temp, rew, util, config_folder = lst
    
    config = 'planning_config' + '_degradation_'+ str(int(contingency_degradation)) \
                      + '_switch_' + str(int(switch_cues))                \
                      + '_train' + str(training_blocks)                   \
                      + '_degr' + str(degradation_blocks)                 \
                      + '_n' + str(trials_per_block)+'.json'


    folder = os.path.join(os.getcwd(),'config/' + config_folder)
    file = open(os.path.join(folder,config))

    task_params = js.load(file)                                                                                 
    colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
    sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
    starts = np.asarray(task_params['starts'])           # starting position of agent
    planets = np.asarray(task_params['planets'])         # planet positions 
    trial_type = np.asarray(task_params['trial_type'])
    blocks = np.asarray(task_params['block'])


    nblocks = int(blocks.max()+1)         # number of blocks
    trials = blocks.size                  # number of trials
    block = task_params['trials_per_block']           # trials per block
 
    meta = {
        'trial_file' : config, 
        'trial_type' : trial_type,
        'switch_cues': switch_cues == True,
        'contingency_degradation' : contingency_degradation == True,
        'learn_rew' : reward_naive == True,
        'context_trans_prob': context_trans_prob,
        'cue_ambiguity' : cue_ambiguity,
        'h' : h,
        'optimal_sequence' : sequence,
        'blocks' : blocks,
        'trials' : trials,
        'nblocks' : nblocks,
        'degradation_blocks': task_params['degradation_blocks'],
        'training_blocks': task_params['training_blocks'],
        'interlace': task_params['interlace'],
        'contingency_degrdataion': task_params['contingency_degradation'],
        'switch_cues': task_params['switch_cues'],
        'trials_per_block': task_params['trials_per_block']
    }

    all_optimal_seqs = np.unique(sequence)                                                                            

    # reward probabilities schedule dependent on trial and planet constelation
    Rho = np.zeros([trials, nr, ns])

    for i, pl in enumerate(planets):
        if i >= block*meta['training_blocks'] and i < block*(meta['training_blocks'] + meta['degradation_blocks']) and contingency_degradation:
            # print(i)
            # print(pl)
            Rho[i,:,:] = planet_reward_probs_switched[tuple([pl])].T
        else:
            Rho[i,:,:] = planet_reward_probs[tuple([pl])].T

    # u = 0.99
    # utility = np.array([(1-u)/2,(1-u)/2,u])

    utility = np.array([float(u)/100 for u in util])
    print(utility)

    if reward_naive==True:
        reward_counts = np.ones([nr, npl, nc])
    else:
        reward_counts = np.tile(planet_reward_probs.T[:,:,np.newaxis]*5,(1,1,nc))+1

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
                1,
                dec_temp,
                rew]

    prefix = 'multiple_'

    if use_fitting == True:
        prefix += 'fitt_'
    else:
        prefix +='hier_'

    if switch_cues == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if contingency_degradation == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'

    fname = prefix +'p' + str(cue_ambiguity) +'_learn_rew' + str(int(reward_naive == True)) + '_q' + str(context_trans_prob) + '_h' + str(h)+ '_' +\
    str(meta['trials_per_block']) +'_'+str(meta['training_blocks']) + str(meta['degradation_blocks']) + \
    '_dec' + str(dec_temp)+ '_rew' + str(rew) + '_u' +  '-'.join(util) + '_' + config_folder
    
    fname +=  '_extinguish.json'

    worlds = [run_agent(par_list, trials, T, ns , na, nr, nc, npl, added=[trial_type,sequence], use_fitting=use_fitting) for _ in range(repetitions)]
    meta['trial_type'] = task_params['trial_type']
    meta['optimal_sequence'] = task_params['sequence']

    worlds.append(meta)

   
    fname = os.path.join(os.path.join(os.getcwd(),'temp'), fname)
    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(worlds)
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)


    return fname


if __name__ == '__main__':


    data_folder = 'temp'


    extinguish = True

    na = 2                                           # number of unique possible actions
    nc = 4                                           # number of contexts, planning and habit
    nr = 3                                           # number of rewards
    ns = 6                                           # number of unique travel locations
    npl = 3
    steps = 3                                        # numbe of decisions made in an episode
    T = steps + 1                                    # episode length


    planet_reward_probs = np.array([[0.95, 0   , 0   ],
                                    [0.05, 0.95, 0.05],
                                    [0,    0.05, 0.95]]).T    # npl x nr
    planet_reward_probs_switched = np.array([[0   , 0    , 0.95],
                                            [0.05, 0.95 , 0.05],
                                            [0.95, 0.05 , 0.0]]).T 
    state_transition_matrix = np.zeros([ns,ns,na])
    m = [1,2,3,4,5,0]
    for r, row in enumerate(state_transition_matrix[:,:,0]):
        row[m[r]] = 1
    j = np.array([5,4,5,6,2,2])-1
    for r, row in enumerate(state_transition_matrix[:,:,1]):
        row[j[r]] = 1
    state_transition_matrix = np.transpose(state_transition_matrix, axes= (1,0,2))
    state_transition_matrix = np.repeat(state_transition_matrix[:,:,:,np.newaxis], repeats=nc, axis=3)


    seed = 5
    np.random.seed(seed)
    ar.manual_seed(seed)

    h =  [1, 2,3,100,200]
    # h = [40]
    cue_ambiguity = [0.9,0.95, 0.98]                       
    context_trans_prob = [0.9, 0.95, 0.98]                
    degradation = [True]
    cue_switch = [False]
    reward_naive = [True]
    training_blocks = [4]
    degradation_blocks=[6,10]
    trials_per_block=[70]
    dec_temps = [1, 2,4,7]
    rew = [0.1]
    utility = [[1, 19, 80], [0.5,0.5,99],[5,25,70],[1, 9, 90]]
    utility = [[1, 9, 90]]

    conf_folder = ['explore']


    # h =  [1,100]
    # # h = [40]
    # cue_ambiguity = [0.9, 0.98]                       
    # context_trans_prob = [0.9]#, 0.98]                
    # degradation = [True]
    # cue_switch = [False]
    # reward_naive = [True,False]
    # training_blocks = [4]
    # degradation_blocks=[6]
    # trials_per_block=[70]
    # dec_temps = [1,4]
    # utility = [[1, 19, 80], [0.5,0.5,99]]#,[5,25,70],[1, 9, 90]]



    arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
            training_blocks, degradation_blocks, trials_per_block,dec_temps, rew\
            , utility,conf_folder]
    use_fitting = False

    repetitions = 1


    lst = []
    path = os.path.join(os.getcwd(),'temp')
    existing_files = os.listdir(path)

    for i in product(*arrays):
        lst.append(list(i))


    names = []

    for li, l in enumerate(lst):
        prefix = 'multiple_'
        if l[0] == True:
            prefix += 'switch1_'
        else:
            prefix +='switch0_'

        if l[1] == True:
            prefix += 'degr1_'
        else:
            prefix += 'degr0_'

        l[11] = [str(entry) for entry in l[11]]
        fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
        str(l[8]) + '_' + str(l[6]) + str(l[7]) + \
        '_dec' + str(l[9]) +'_rew' + str(l[10]) + '_' + 'u'+  '-'.join(l[11]) + '_' + l[12]

        if extinguish:
            fname += '_extinguish.json'
        else:
            fname += '.json'
        names.append([li, fname])


    # missing_files = []
    # for name in names:
    #     if not name[1] in existing_files:
    #         # print(name)
    #         missing_files.append(name[0])

    # lst = [lst[i] for i in missing_files]
    print('simulations to run: ' + str(len(lst)))

    ca = [ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs,\
        planet_reward_probs_switched,repetitions,use_fitting]

    # if True:
    if True:
        print(names[0])
        for l in [lst[0]]:
            run_single_sim(l,
                            ca[0],\
                            ca[1],\
                            ca[2],\
                            ca[3],\
                            ca[4],\
                            ca[5],\
                            ca[6],\
                            ca[7],\
                            ca[8],\
                            ca[9],\
                            ca[10])

    with Pool() as pool:

        for _ in tqdm.tqdm(pool.istarmap(run_single_sim, zip(lst,\
                                                repeat(ca[0]),\
                                                repeat(ca[1]),\
                                                repeat(ca[2]),\
                                                repeat(ca[3]),\
                                                repeat(ca[4]),\
                                                repeat(ca[5]),\
                                                repeat(ca[6]),\
                                                repeat(ca[7]),\
                                                repeat(ca[8]),\
                                                repeat(ca[9]),\
                                                repeat(ca[10]))),
                        total=len(lst)):
            pass



