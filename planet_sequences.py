#%%
import pandas as pd
import numpy as np
from itertools import combinations, combinations_with_replacement, permutations, product
import json

#%%

'''
deal only with storing individual 
configuations optimal for eahc sequence
'''

# def export_to_json(planets, starts, i ):
#     config = {'starts' : starts,'planets': planets}

#     # str = json.dumps(config)
#     # str = "'[" + str + "]'"  
#     # with open("config_task.json", "w") as file:
# 	#     json.dump(str, file)
#     with open("config_task" + str(i) +".json", "w") as file:
# 	    json.dump(config, file)

# slices = []
# for i in range(8):
#     slice = data.loc[( data['optimal'] == True) & ( data['sequence'] == i)]
#     slices.append(slice)
#     print(i, slice.shape[0])
#     datatype = {'conf_ind':int, 'start':int, 'planet_conf':int, 'sequence':int}
#     slice = slice.astype(datatype)

#     plnts = slice['planet_conf']
#     plnts = planet_confs[[plnts.values.tolist()]].tolist()
#     strts = slice['start'].values.tolist()
#     optimal_seq = i

#     export_to_json(plnts, strts,i)

''' 
creates all permutations of a vector r which holds possible digits
for r = [0,1] and length of sequence n = 3
000 001 010 011 100 101 etc 
'''
def sequence_of_length_n(r,n):
    r = [r]*n
    return np.asarray(list(product(*r)))


'''
function calculating the expected reward for a given planet constelation and action sequence

p = reward probabilities for the planets
conf = planet configuration
stm = state transition matrix
sequence = action sequence such as jump jump move
rewards = possible rewards a planet can bring
'''

def create_path(conf, p, stm, sequence, r):

    probs = np.array([[ p[planet] for planet in conf]])           # extract reward probabilities for planets
    probs = np.repeat(probs, repeats = stm.shape[0], axis=0)

    
    rewards = np.array([[ r[planet] for planet in conf]])           # extract reward probabilities for planets
    rewards = np.repeat(rewards, repeats = stm.shape[0], axis=0)

    # path holds position of rocket at each time step
    # different rows correspond to different starting positions
    path = np.zeros([stm.shape[0], stm.shape[1], 3])
    path[:,:,0] = stm[:,:,sequence[0]]
    
    for step in range(2):
        path[:,:,step+1] = path[:,:,step].dot(stm[:,:,sequence[step+1]])

    expectation = np.zeros([stm.shape[0], stm.shape[1]])
    for step in range(3):
        expectation += path[:,:,step]*probs            
    
    expectation = expectation*rewards
    expectation = expectation.sum(axis=1)

    return expectation


#%%
def generate_trials_df(planet_rewards, sequences):

    # create state transition matrices
    nplanets = 6
    state_transition_matrix = np.zeros([6,6,2])

    m = [1,2,3,4,5,0]
    for r, row in enumerate(state_transition_matrix[:,:,0]):
        row[m[r]] = 1

    j = np.array([5,4,5,6,2,2])-1
    for r, row in enumerate(state_transition_matrix[:,:,1]):
        row[j[r]] = 1



    # reward probabiltiy vector
    p = [0.95, 0.05, 0.95]
    # planet_rewards = [-1,1,1]
    moves = sequence_of_length_n([0,1],3)
    planet_confs = sequence_of_length_n([0,1,2],6)

    # generate all planet constelations
    planet_confs = np.delete(planet_confs,[0,planet_confs.shape[0]-1], 0)
    starts = np.tile(np.arange(nplanets), planet_confs.shape[0])


    expectations = np.zeros([planet_confs.shape[1], moves.shape[0],  planet_confs.shape[0]])


    row_data = {

        'conf_ind': 0,
        'planet_conf':0,
        'start': 0,
        'sequence': 0,
        'p0': 0,
        'p1': 0,
        'p2': 0,
        'p3': 0,
        'p4': 0,
        'p5': 0,
        'expected_reward': 0
    }

    data = pd.DataFrame(columns = row_data.keys(), index = np.arange(planet_confs.shape[0]*nplanets*moves.shape[0]))


    for ci, conf in enumerate(planet_confs):
        for m, move in enumerate(moves):
            expectations[:, m, ci] = create_path(conf, p, state_transition_matrix, move, planet_rewards)

    s = 0
    i = 0
    for ci, conf in enumerate(planet_confs):
        for st in np.arange(nplanets):
            for m, move in enumerate(moves):
                rd = [s, ci, st, m, conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], expectations[st,m,ci]]
                k = 0
                for key in row_data:
                    row_data[key] = rd[k]
                    k += 1
                k = 0
                data.loc[i] = pd.Series(row_data)
                i += 1
            s += 1


    # # define max reward for a given conformation and starting point
    data['max_reward'] = data.groupby('conf_ind')[['expected_reward']].transform('max')

    # round respected entries
    data['max_reward'] = data['max_reward'].astype(float).round(3)
    data['expected_reward'] = data['expected_reward'].astype(float).round(3)


    # check rounding was sucessfull cause apparently it is a fucking nightmare to round > _ >
    print(data[data['conf_ind'] == 93]['max_reward'].tolist())
    print(data[data['conf_ind'] == 93]['expected_reward'].tolist())

    # define optimal sequnces
    data['optimal'] = data['max_reward'] == data['expected_reward']
    # count optimal sequences
    data['total_optimal'] = data.groupby(['conf_ind'])[['optimal']].transform('sum')

    # drop all configurations that have more than 1 optimal sequence
    data = data.drop(data[data.total_optimal != 1].index)



    '''
    IF ONE WANTS TO DISCARD TRIALS WITH NEXT 
    BEST SEQUENCE GIVING EXPECTED REWARD LOWER
    THAN CUT OFF LEVEL X
    '''
    # x = 1
    # nseqs = moves.shape[0]
    # data['diff'] = data['max_reward'] - data['expected_reward']
    # data = data.sort_values(by = ['conf_ind', 'diff'])
    # data['diff_order'] = np.tile(np.arange(nseqs),np.int32(data.shape[0]/nseqs))
    # small_difference = data.loc[(data['diff_order'] == 1) & (data['diff'] < x)]
    # inds = np.unique(small_difference['conf_ind'])
    # data = data[~data.conf_ind.isin(inds)]
    # data.head(20)




    '''
    Generate trials for the actual experiment
    '''

    s1 = sequences[0]       # optimal sequence 1
    s2 = sequences[1]       # optimal sequence 2

    slices = []

    for s in [s1, s2]:
        slice = data.loc[( data['optimal'] == True) & ( data['sequence'] == s)]
        datatype = {'conf_ind':int, 'start':int, 'planet_conf':int, 'sequence':int}
        slice = slice.astype(datatype)
        slices.append(slice)

    return slices, planet_confs

    # n1 = slices[0].shape[0]      # number of trials for that sequence
    # n2 = slices[1].shape[0]      # number of trials for that sequence

    # dt = np.zeros([n1 + n2, 3 + nplanets])  # data storage array

    # plnts = slices[0].planet_conf
    # plnts = planet_confs[[plnts.values.tolist()]].tolist()
    # strts = slices[0].start.values.tolist()

    # dt[0:n1,0] = [0]*n1      # context index
    # dt[0:n1,1] = [s1]*n1     # optimal sequence index
    # dt[0:n1,2] = strts       # trial starting position
    # dt[0:n1,3:] = plnts     

    # plnts = slices[1].planet_conf
    # plnts = planet_confs[[plnts.values.tolist()]].tolist()
    # strts = slices[1].start.values.tolist()

    # dt[n1:n1+n2,0] = [1]*n2
    # dt[n1:n1+n2,1] = [s2]*n2
    # dt[n1:n1+n2,2] = strts
    # dt[n1:n1+n2,3:] = plnts

    # np.random.shuffle(dt)
    # dt = dt.astype('int32')


    # # export to json
    # config = {'context' : dt[:,0].tolist(),
    #           'sequence': dt[:,1].tolist(),
    #           'starts': dt[:,2].tolist(),
    #           'planets': dt[:,3:].tolist()
    #           }

    # with open("config_task.json", "w") as file:
    #     json.dump(config, file)

# %%


