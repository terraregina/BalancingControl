#%%
import numpy as np
import pandas as pd
import json
import sys

fname = 'config/shuffled_and_blocked/planning_config_degradation_1_switch_0_train6_degr6_n42.json'
if sys.platform == 'win32':
   fname = fname.replace('/', '\\') 
f = open(fname)

data = json.load(f)
context_coding = np.array(["habit_training", "planning"])
planet_coding = np.array([1,3,5]) # 1 = red, 3 = gray, 5 = green;
planet_coding[data['planets']]
# [1,6]
# planetRewardProbs
# planetRewardProbsExtinction
# stateTransitionMatrix

config = {
    'conditionsExp': {
        'planets': planet_coding[data['planets']].tolist(),
        'starts': (np.array(data['starts']) + 1).tolist(),
        'contexts': context_coding[data['context']].tolist(),
        'trial_type':data['trial_type']
    },
    # update for training conditions
    'conditionsTrain': {
        'planets': planet_coding[data['planets']].tolist(),
        'starts': (np.array(data['starts']) + 1).tolist(),
        'contexts': context_coding[data['context']].tolist(),
        'trial_type':data['trial_type']
    },
    'planetRewardProbs':0,
    'planetRewardProbsExtinction':0,
    'planetRewards': [-1,0,1],
    'stateTransitionMatrix' : 0
    
}
# %%
