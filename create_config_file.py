#%%
import numpy as np
import pandas as pd
import json
import sys


# hardcoded - not good;
planetRewardProbs = [[0.9, 0, 0   , 0, 0   ],
                        [0.1, 0, 0.9, 0, 0.1 ],
                        [0   , 0, 0.1, 0, 0.9]]
                        
planetRewardProbsDegradation = [[0,    0, 0.1, 0, 0.9],
                                [0.1, 0, 0.9, 0, 0.1],
                                [0.9, 0,    0, 0,    0]]

context_coding = np.array(["habit_training", "planning"])
planet_coding = np.array([1,3,5]) # 1 = red, 3 = gray, 5 = green;
# # [1,6]

# # planetRewardProbs
# # planetRewardProbsExtinction
# # stateTransitionMatrix

####################################################################################################

day2 = 'config/shuffled_and_blocked/new_planning_config_degradation_1_switch_0_train2_degr2_n42_nr_3.json'
day1 = 'config/shuffled_and_blocked/new_planning_config_degradation_1_switch_0_train5_degr0_n42_nr_3.json'

files = {
         'day1' : day1,
         'day2' : day2
        }
debugging = False
data_array = []
for key in files:
    f = files[key]

    if sys.platform == 'win32':
        f = f.replace('/', '\\') 
        f = open(f)

    data = json.load(f)
    
    data_array.append(data)
    if debugging:
        df = pd.DataFrame.from_dict(data)
        if key == 'day2': 
            a = [0,1,2,3,4,28,29,30,31,32,56,57,58,59,60]
        else:
            a = np.arange(28)
        df = df.loc[a]
        data = df.to_dict(orient='list')

    config = {
        'conditionsExp': {
            'planets': planet_coding[np.array(data['planets'])].tolist(),
            'starts': (np.array(data['starts']) + 1).tolist(),
            'contexts': context_coding[data['context']].tolist(),
            'trial_type':data['trial_type'],
            'optimal_sequence': data['sequence']
        },
        # update for training conditions
 
        'planetRewardProbs':planetRewardProbs,
        'planetRewardProbsDegradation':planetRewardProbsDegradation,
        'planetRewards': [-1,0,1],
        'stateTransitionMatrix' : data['state_transition_matrix'][0]     
    }

    if key == 'day1':

        # create training data
        n_train_trials = 20
        df = pd.DataFrame.from_dict(data)


        df = df.query('context != 0 & trial_type == 0').sample(n_train_trials)
        # print(df.head(20))
        data = df.to_dict(orient='list')

        config['conditionsTrain'] =  {
                'planets': planet_coding[np.array(data['planets'])].tolist(),
                'starts': (np.array(data['starts']) + 1).tolist(),
                'contexts': context_coding[data['context']].tolist(),
                'trial_type':[-1]*n_train_trials,
                'optimal_sequence':data['sequence']
            }
    


    # save file config file

    with open(key + '.json', 'w') as outfile:
        json.dump(config, outfile)
# %%

dfs = []
for data in data_array:
    dfs.append(pd.DataFrame.from_dict(data))
df = pd.concat(dfs).reset_index(drop=True)
blocks_total = df.query('training_blocks == 5').block.unique().max()+1
blocks_total += df.query('training_blocks == 2').block.unique().max()+1
df['training_blocks'] = 7
df['degradation_blocks'] = 2
df['block'] = np.arange(blocks_total).repeat(df['trials_per_block'].unique()[0])
cols = df.columns
config = {}
for col in cols:
    config[col] = df[col].tolist()

name = 'config/shuffled_and_blocked/new_planning_config_degradation_1_switch_0_train7_degr2_n42_nr_3.json'
with open(name, 'w') as outfile:
    json.dump(config, outfile)