# %%
# Imports
# 
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
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from itertools import product, repeat
import os
import action_selection as asl
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
import time
import time
from multiprocessing import Pool
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

data_folder = 'data'

# %%
# Run simulations

from run_exampe_habit_v1 import run_agent, run_single_sim

na = 2                                           # number of unique possible actionsprint(len(lst))
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
np.random.seed()

h =  [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,100]
cue_ambiguity = [0.5, 0.6, 0.7,0.8]                       
context_trans_prob = [1/nc -0.1, 1/nc, 1/nc+0.1]                
degradation = [False,True]
cue_switch = [True, False]
reward_naive = [False, True]
training_blocks = [4]
degradation_blocks=[2,4,6]
trials_per_block=[60]
arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
        training_blocks, degradation_blocks, trials_per_block]


repetitions = 5
lst = []
path = os.path.join(os.getcwd(),'data')
existing_files = os.listdir(path)

for i in product(*arrays):
    lst.append(list(i))


names = []


for li, l in enumerate(lst):
    prefix = ''
    if l[0] == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if l[1] == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'

    fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
    str(l[8]) + '_' + str(l[6]) + str(l[7]) + '.json'
    names.append([li, fname])

existing_files = os.listdir(path)
path = os.path.join(os.getcwd(),data_folder)

missing_files = []
for name in names:
    if not name[1] in existing_files:
        missing_files.append(name[0])

lst = [lst[i] for i in missing_files]
print('simulations to run: ' + str(len(lst)))

ca = [ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs,\
    planet_reward_probs_switched,repetitions]

simulate = True

if simulate:
    # start =  time.perf_counter()

    # with Pool() as pool:
    #     M = pool.starmap(run_single_sim, zip(lst,\
    #                                         repeat(ca[0]),\
    #                                         repeat(ca[1]),\
    #                                         repeat(ca[2]),\
    #                                         repeat(ca[3]),\
    #                                         repeat(ca[4]),\
    #                                         repeat(ca[5]),\
    #                                         repeat(ca[6]),\
    #                                         repeat(ca[7]),\
    #                                         repeat(ca[8]),\
    #                                         repeat(ca[9])))
    # print(len(M))
    # finish = time.perf_counter()
    # print(f'Finished in {round(finish-start, 2)} second(s) for multithread')

 


    with Pool() as pool:
            # iterables = zip(lst, repeat(ca[0]), repeat(ca[1]), repeat(ca[2]), repeat(ca[3]), repeat(ca[4]), repeat(ca[5]), repeat(ca[6]), repeat(ca[7]), repeat(ca[8]), repeat(ca[9]))

        for _ in tqdm.tqdm(pool.istarmap(run_single_sim, zip(lst[50:],\
                                                repeat(ca[0]),\
                                                repeat(ca[1]),\
                                                repeat(ca[2]),\
                                                repeat(ca[3]),\
                                                repeat(ca[4]),\
                                                repeat(ca[5]),\
                                                repeat(ca[6]),\
                                                repeat(ca[7]),\
                                                repeat(ca[8]),\
                                                repeat(ca[9]))),
                        total=len(lst)):
            pass



# %%



