#%%
import numpy as np
import pandas as pd
import json
import sys


# hardcoded - not good;
planetRewardProbs = [[0.95, 0, 0   , 0, 0   ],
                        [0.05, 0, 0.95, 0, 0.05 ],
                        [0   , 0, 0.05, 0, 0.95]]
                        
planetRewardProbsDegradation = [[0,    0, 0.05, 0, 0.95],
                                [0.05, 0, 0.95, 0, 0.05],
                                [0.95, 0,    0, 0,    0]]

context_coding = np.array(["habit_training", "planning"])
planet_coding = np.array([1,3,5]) # 1 = red, 3 = gray, 5 = green;
# # [1,6]

# # planetRewardProbs
# # planetRewardProbsExtinction
# # stateTransitionMatrix

####################################################################################################

day2 = 'config/shuffled_and_blocked/planning_config_degradation_1_switch_0_train1_degr1_n28_nr_3.json'
day1 = 'config/shuffled_and_blocked/planning_config_degradation_1_switch_0_train2_degr0_n28_nr_3.json'
# train = day1

files = {
        #  'train': train,
         'day1' : day1,
         'day2' : day2
        }

for key in files:
    f = files[key]

    if sys.platform == 'win32':
        f = f.replace('/', '\\') 
        f = open(f)

    data = json.load(f)
    


    config = {
        'conditionsExp': {
            'planets': planet_coding[data['planets']].tolist(),
            'starts': (np.array(data['starts']) + 1).tolist(),
            'contexts': context_coding[data['context']].tolist(),
            'trial_type':data['trial_type'],
            'optimal_sequence':data['sequence']
        },
        # update for training conditions

        'planetRewardProbs':planetRewardProbs,
        'planetRewardProbsExtinction':planetRewardProbsDegradation,
        'planetRewards': [-1,0,1],
        'stateTransitionMatrix' : 0     
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
