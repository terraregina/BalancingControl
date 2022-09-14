#%%                     
import json  as js
import numpy as np
from operator import truediv
import numpy as np
import os
from itertools import product, repeat
import pickle
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
# from planet_sequences import generate_trials_df
import pandas as pd
from itertools import combinations, combinations_with_replacement, permutations, product
import json
import pandas as pd 


#%%

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

    slices = np.empty(len(sequences), dtype=object)

    for si, s in enumerate(sequences):
        slice = data.loc[( data['optimal'] == True) & ( data['sequence'] == s)]
        datatype = {'conf_ind':int, 'start':int, 'planet_conf':int, 'sequence':int, 'expected_reward':float}
        slice = slice.astype(datatype)
        # print(slice.columns)
        slices[si] = slice
        # print(slices[0][0])
    return slices, planet_confs


def all_possible_trials(habit_seq=3, shuffle=False, extend=False):
    
    np.random.seed(1)
    ns=6
    all_rewards = [[-1,1,1], [1,1,-1]]
    sequences  = np.arange(8)
    
    slices = [None for nn in range(2)]
    
    planet_confs = [None for nn in range(2)]
    
    for ri, rewards in enumerate(all_rewards):
        slices[ri], planet_confs[ri] = generate_trials_df(rewards, sequences)


    data = [[] for i in range(len(sequences))]

    for s in sequences:
        for ci, trials_for_given_contingency in enumerate(slices):
            slice = trials_for_given_contingency[s]
            planet_conf = planet_confs[ci]
            ntrials = slice.shape[0]
            dt = np.zeros([slice.shape[0], 3 + ns])
            plnts = slice.planet_conf                             # configuration indeces for s1 trials
            plnts = planet_conf[[plnts.values.tolist()]].tolist() # actual planet configurations for s1 trials
            strts = slice.start.values.tolist()                   # starting points for s1 trials 
            expected_reward = slice.expected_reward.values.tolist()  
            # dt[:,0] = [0]*ntrials      # context index
            dt[:,0] = [s]*ntrials        # optimal sequence index
            dt[:,1] = strts              # trial starting position
            dt[:,2:-1] = plnts           # planets
            dt[:,-1] = expected_reward   # planets
            if shuffle:
                np.random.shuffle(dt)
            data[s].append(dt)


    if extend:
        data_extended = [[] for _ in range(len(data))]
        reps = 5

        for di, dat in enumerate(data):
            for contingency in range(2):
                if di == habit_seq:
                    context_cue = np.zeros([dat[contingency].shape[0]*reps,1])
                else:
                    context_cue = np.ones([dat[contingency].shape[0]*reps,1])

                extended =  np.hstack((context_cue, np.tile(dat[contingency], (reps,1))))   
                data_extended[di].append(extended)
    else:
        data_extended = data.copy()
    
    return data_extended


def create_trials_planning(data, habit_seq = 3, contingency_degradation = True,\
                        switch_cues= False,\
                        training_blocks = 2,\
                        degradation_blocks = 1,\
                        extinction_blocks = 2,\
                        interlace = True,\
                        block = None,\
                        trials_per_block = 28, export = True,blocked=False,shuffle=False,seed=1):

    np.random.seed(seed)

    sequences = np.arange(8,dtype=int)

    if trials_per_block % 2 != 0:
        raise Exception('Give even number of trials per block!')

    nblocks = training_blocks + degradation_blocks + extinction_blocks
    half_block = np.int(trials_per_block/2)
    ntrials = nblocks*trials_per_block//2
    ncols = data[0][0].shape[1]
    # block = np.int(trials[0].shape[0]/nblocks)
    trials = np.zeros([nblocks*trials_per_block, ncols])
    trial_type = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)
    n_planning_seqs = sequences.size -1
    planning_seqs = sequences[sequences != habit_seq]
    shift = half_block//n_planning_seqs
    tt=0

    split_data = [[None for c in range(2)] for s in sequences]

    # I am takign this one as a reference since this occurs in all of them
    # sequence 4 and 6 have actually combinations with expected rewards of -1.85 and -0.95
    unique = np.unique(data[0][0][:,-1])
    probs = (np.arange(unique.size)+1)/(np.arange(unique.size)+1).sum()
    rewards = np.random.choice(np.arange(unique.size), p=probs, size=ntrials)
    planning_seqs = sequences[sequences != habit_seq]

    for s in range(len(data)):
        for c in range(len(data[s])):
            print(unique)
            dat = data[s][c]
            split_data[s][c] = [dat[dat[:,-1]==k] for k in unique]
            np.random.shuffle(split_data[s][c])
    

    for i in range(nblocks):
        if i == (nblocks - extinction_blocks):                   # if now populating extinction block
            tt = 2

        if i >= training_blocks and i < training_blocks + degradation_blocks:
            contingency = 1
        else: 
            contingency = 0 
        pl_seq_ind = -1
        for t in range(0,trials_per_block,2):
            print(t)
            tr_h = i*trials_per_block + t
            tr_p = i*trials_per_block + t + 1
            pl_seq_ind += 1
            data[tr_h,:] = split_data[habit_seq][contingency][0]
            data[tr_p,:] = split_data[pl_seq_ind][contingency][0]

            if pl_seq_ind == n_planning_seqs - 1:
                pl_seq_ind = -1
             
    # if not blocked:    
    #     # populate training blocks and extinction block
    #     for i in range(nblocks):
    #         if i == (nblocks -2):                   # if now populating extinction block
    #             tt = 2

    #         trials[(2*i)*half_block : (2*i+1)*half_block , :] = data[habit_seq][0][i*half_block : (i+1)*half_block,:]
    #         planning_trials = np.zeros([half_block, ncols])
    #         for si, s in enumerate(planning_seqs):
    #             planning_trials[si*shift:(si+1)*shift,:] = data[s][0][i*shift:(i+1)*shift,:]           
    #         trials[(2*i+1)*half_block : (2*i+2)*half_block , :] = planning_trials

    #         trial_type[(2*i)*half_block:(2*i+2)*half_block] = tt
    #         blocks[(2*i)*half_block:(2*i+2)*half_block] = i
            

    #     # populate the degradation blocks
    #     for i in range(training_blocks, degradation_blocks + training_blocks):
    #         trial_type[(2*i)*half_block:(2*i+2)*half_block] = 1

    #         if contingency_degradation:
    #             ind = 1
    #         else:
    #             ind = 0

    #         trials[(2*i)*half_block : (2*i+1)*half_block , :] = data[habit_seq][ind][i*half_block:(i+1)*half_block,:]
    #         planning_trials = np.zeros([half_block, ncols])
    #         for si, s in enumerate(planning_seqs):
    #             planning_trials[si*shift:(si+1)*shift,:] = data[s][ind][i*shift:(i+1)*shift,:]
    #         trials[(2*i+1)*half_block : (2*i+2)*half_block , :] = planning_trials

    #         if switch_cues:
    #             trials[:,0] = trials[:,0] == 0

    #     if interlace:
    #         for i in range(nblocks):
    #             np.random.shuffle(trials[(2*i)*half_block:(2*i+2)*half_block,:])
                

    # elif blocked:
    #     miniblocks = half_block//block
        
    #         # populate training blocks and extinction block
    #     for i in range(nblocks):

    #         if i >= training_blocks and i < training_blocks + degradation_blocks:
    #             ind = 1
    #             tt = 1
    #         else: 
    #             ind = 0
    #             tt = 0

    #         if i >= (nblocks -2):                   # if now populating extinction block
    #             tt = 2

    #         planning_trials = np.zeros([half_block, ncols])
    #         for si, s in enumerate(planning_seqs):
    #             planning_trials[si*shift:(si+1)*shift,:] = data[s][ind][i*shift:(i+1)*shift,:]

    #         np.random.shuffle(planning_trials)         

    #         for mb in range(miniblocks):
    #             start_trials_habit = i*trials_per_block + (2*mb)*block
    #             stop_trials_habit =  i*trials_per_block + (2*mb+1)*block
    #             stop_trials_planning =  i*trials_per_block + (2*mb+2)*block
    #             start_data =  i*trials_per_block +  mb*block
    #             stop_data =  i*trials_per_block +  (mb+1)*block

    #             trials[start_trials_habit : stop_trials_habit , :] = data[habit_seq][ind][start_data : stop_data,:]
    #             trials[stop_trials_habit : stop_trials_planning , :] = planning_trials[mb*block : (mb+1)*block,:]

    #             # trials[(2*mb)*block : (2*mb+1)*block , :] = data[habit_seq][0][mb*block : (mb+1)*block,:]
    #             # trials[(2*mb+1)*block : (2*mb+2)*block , :] = planning_trials[mb*block : (mb+1)*block,:]

    #         trial_type[(2*i)*half_block:(2*i+2)*half_block] = tt
    #         blocks[(2*i)*half_block:(2*i+2)*half_block] = i
            

    # # trials[:,:-1] = trials[:,:-1].astype('int32')
    # trial_type = trial_type.astype('int32')

    # fname = 'planning_config_'+'degradation_'+ str(int(contingency_degradation))+ '_switch_' + str(int(switch_cues))\
    #          + '_train' + str(training_blocks) + '_degr' + str(degradation_blocks) + '_n' + str(trials_per_block)+'.json'
    # fname = 'test.json'


    # if shuffle and blocked:
    #     subfolder = '/shuffled_and_blocked'
    # else:
    #     subfolder ='/shuffled'
    # path = os.path.join(os.getcwd(),'config'+subfolder)
    # fname = os.path.join(path, fname)

    # if export:
    #     config = {
    #               'context' : trials[:,0].astype('int32').tolist(),
    #               'sequence': trials[:,1].astype('int32').tolist(),
    #               'starts': trials[:,2].astype('int32').tolist(),
    #               'planets': trials[:,3:-1].astype('int32').tolist(),
    #               'exp_reward': trials[:,-1].tolist(),
    #               'trial_type': trial_type.tolist(),
    #               'block': blocks.tolist(),
    #               'degradation_blocks': degradation_blocks,
    #               'training_blocks': training_blocks,
    #               'interlace': interlace,
    #               'contingency_degradation': contingency_degradation,
    #               'switch_cues': switch_cues,
    #               'trials_per_block': trials_per_block,
    #               'blocked': blocked,
    #               'shuffle': shuffle, 
    #               'miniblock_size' : block,
    #               'seed':seed
    #             }

    #     with open(fname, "w") as file:
    #         js.dump(config, file)


def create_config_files_planning(training_blocks, degradation_blocks, trials_per_block, habit_seq = 3,\
    shuffle=False, blocked=False, block=None):
    trials = all_possible_trials(habit_seq=habit_seq,shuffle=shuffle)
    degradation = [True]
    cue_switch = [False]
    arrays = [degradation, cue_switch, degradation_blocks, training_blocks, trials_per_block,[blocked]]

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:
        print(l)
        create_trials_planning(trials, contingency_degradation=l[0],switch_cues=l[1],degradation_blocks=l[2],
                            training_blocks=l[3], trials_per_block=l[4], habit_seq = habit_seq,\
                            blocked = l[-1],shuffle=shuffle,block=block)





conf = ['shuffled', 'shuffled_and_blocked', 'blocked','original']
data_folder='config'


for con in conf:
    path = os.path.join(data_folder, con)

    if not os.path.exists(path):
        os.makedirs(path)




combinations = []
create_config_files_planning([3],[1],[70],shuffle=True)

print(np.array[(1,2,3)])































# create_config_files([4], [24,6], [70])
# create_config_files_planning([4],[1],[70],shuffle=True)
# create_config_files_planning([4],[6],[70],shuffle=True,blocked=True, block=7)





# trials = all_possible_trials_two_seqs(shuffle=False)
# trials_shuffled = all_possible_tarials_two_seqs(shuffle=True)


# create_config_files_planning([2],[1],[70],shuffle=True, blocked=True, block=5)
# create_config_files([3],[1],[70],shuffle=False,blocked=False)
# create_config_files([2],[1],[70],shuffle=True, blocked = True, block = 5)


# fname = 'config/' + 'config_degradation_1_switch_0_train4_degr2_n70.json' 
# file = os.path.join(os.getcwd(), fname)
# import pandas as pd
# file = open(file, 'r')
# data = js.load(file)
# df = pd.DataFrame(data)
# df.query('trial_type == 2').tail(30)


# %%
# import json 
# import pandas as pd


# f = open('config/shuffled/config_degradation_0_switch_0_train1_degr1_n20.json')
# data = json.load(f)
# df = pd.DataFrame.from_dict(data)
# df.head(50)


#%%

# trials = all_possible_trials_planning()
# %%
