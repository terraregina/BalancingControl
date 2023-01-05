#%% IMPORTS
import json  as js
from re import L
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


#%% FUNCTIONS

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

    probs = p[conf]   # extract reward probabilities for planets
    probs = np.repeat(probs.reshape(1, probs.shape[0]), repeats = stm.shape[0], axis=0)

    
    # rewards = np.array([[ r[planet] for planet in conf]])           # extract reward probabilities for planets
    rewards = r[conf]
    rewards = np.repeat(rewards.reshape(1, rewards.shape[0]), repeats = stm.shape[0], axis=0)

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
    if len(planet_rewards) == 3:
        p = np.array([0.95, 0.05, 0.95])
        planet_confs = sequence_of_length_n([0,1,2],6)
    else:
        p = np.array([0.95, 0.95])
        planet_confs = sequence_of_length_n([0,1],6)

    # planet_rewards = [-1,1,1]
    moves = sequence_of_length_n([0,1],3)

    # generate all planet constelations
    planet_confs = np.delete(planet_confs,[0,planet_confs.shape[0]-1], 0)
    starts = np.tile(np.arange(nplanets), planet_confs.shape[0])


    expectations = np.zeros([planet_confs.shape[1], moves.shape[0],  planet_confs.shape[0]])

    col_names = ['conf_ind', 'planet_conf', 'start', 'sequence', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'expected_reward', ]


    for ci, conf in enumerate(planet_confs):
        for m, move in enumerate(moves):

            expectations[:, m, ci] = create_path(conf, p, state_transition_matrix, move, planet_rewards)

    s = 0
    i = 0
    rr = 0
    nrows = planet_confs.shape[0]*nplanets*moves.size 
    data_array = np.zeros([nrows, len(col_names)])
    for ci, conf in enumerate(planet_confs):
        for st in np.arange(nplanets):
            for m, move in enumerate(moves):
                data_array[rr,:] = \
                [s, ci, st, m, conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], expectations[st,m,ci]]
                # rd = [s, ci, st, m, conf[0], conf[1], conf[2], conf[3], conf[4], conf[5], expectations[st,m,ci]]
                # k = 0
                # for key in row_data:
                #     row_data[key] = rd[k]
                #     k += 1
                # k = 0
                rr += 1
                # data.loc[i] = pd.Series(row_data)
                i += 1
            s += 1
    data = pd.DataFrame(data_array, columns = col_names)


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
    return slices, planet_confs,state_transition_matrix

def all_possible_trials(habit_seq=3, shuffle=False, extend=False, nr=3):
    
    np.random.seed(1)
    ns=6

    if nr == 3:
        all_rewards = np.array([[-1,1,1], [1,1,-1]])
    elif nr == 2:
        all_rewards = np.array([[-1,1], [1,-1]])
    else:
        print('Rewards are wrong')
    sequences  = np.arange(8)
    
    slices = [None for nn in range(2)]
    
    planet_confs = [None for nn in range(2)]
    
    for ri, rewards in enumerate(all_rewards):
        slices[ri], planet_confs[ri],state_transition_matrix = generate_trials_df(rewards, sequences)


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
    
    return data_extended, state_transition_matrix


def create_trials_planning(data, habit_seq = 3, contingency_degradation = True,\
                        switch_cues= False,\
                        training_blocks = 2,\
                        degradation_blocks = 1,\
                        extinction_blocks = 2,\
                        interlace = True,\
                        block = None,\
                        trials_per_block = 28,\
                        export = True,blocked=False,shuffle=False,\
                        nr= 3, seed=1,stm=None):

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
    context = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)
    n_planning_seqs = sequences.size -1
    shift = half_block//n_planning_seqs
    tt=0

    split_data = [[None for c in range(2)] for s in sequences]

    # I am takign this one as a reference since this occurs in all of them
    # sequence 4 and 6 have actually combinations with expected rewards of -1.85 and -0.95
    unique = np.unique(data[0][0][:,-1])
    probs = (np.arange(unique.size)+1)/(np.arange(unique.size)+1).sum()
    rewards = np.random.choice(np.arange(unique.size), p=probs, size=ntrials)
    planning_seqs = sequences[sequences != habit_seq]
    planning_seqs = np.tile(planning_seqs, half_block // planning_seqs.size)

    for s in range(len(data)):
        for c in range(len(data[s])):
            print(unique)
            dat = data[s][c]
            split_data[s][c] = [dat[dat[:,-1]==k] for k in unique]
            for k in range(unique.size):
                np.random.shuffle(split_data[s][c][k])
                pass
    
    if block is not None:
        miniblocks = half_block//block



    for i in range(nblocks):

        if i >= training_blocks and i < training_blocks + degradation_blocks:
            contingency = 1
            tt = 1
        else: 
            contingency = 0
            tt = 0 

        if i >= (nblocks - extinction_blocks): 
            tt = 2

        if not blocked:

            # populate habit seq trials
            for t in range(half_block):
                tr = int(i*trials_per_block + t)
                ri = half_block*i + t
                trials[tr,:] = split_data[habit_seq][contingency][rewards[ri]][0]
                split_data[habit_seq][contingency][rewards[ri]] = \
                    np.roll(split_data[habit_seq][contingency][rewards[ri]], -1, axis=0)

            # populate planning trials
            for t in range(half_block):
                tr = int(i*trials_per_block + half_block + t)
                ri = half_block*i + t
                trials[tr,:] = split_data[planning_seqs[t]][contingency][rewards[ri]][0]
                split_data[planning_seqs[t]][contingency][rewards[ri]] = \
                    np.roll(split_data[planning_seqs[t]][contingency][rewards[ri]], -1, axis=0)

            np.random.shuffle(trials[i*trials_per_block:(i+1)*trials_per_block,:])

        if blocked:

            # extract all planning trials and shuffle them
            habit_trials = np.zeros([half_block, trials.shape[1]])
            planning_trials = np.zeros([half_block, trials.shape[1]])
            
            for t in range(half_block):
                tr = int(i*trials_per_block + half_block + t)
                ri = half_block*i + t

                habit_trials[t,:] = split_data[habit_seq][contingency][rewards[ri]][0]
                split_data[habit_seq][contingency][rewards[ri]] = \
                    np.roll(split_data[habit_seq][contingency][rewards[ri]], -1, axis=0)
            

                planning_trials[t,:] = split_data[planning_seqs[t]][contingency][rewards[ri]][0]
                split_data[planning_seqs[t]][contingency][rewards[ri]] = \
                    np.roll(split_data[planning_seqs[t]][contingency][rewards[ri]], -1, axis=0)
            np.random.shuffle(planning_trials)

            for mb in range(miniblocks):
                tr = int(i*trials_per_block)
                trials[tr+2*mb*block:tr+(2*mb+1)*block] = habit_trials[mb*block:(mb+1)*block,:]
                trials[tr+(2*mb+1)*block:tr+(2*mb+2)*block] = planning_trials[mb*block:(mb+1)*block,:]
            

        trial_type[i*trials_per_block:(i+1)*trials_per_block] = int(tt)
        blocks[i*trials_per_block:(i+1)*trials_per_block] = int(i)
        context[i*trials_per_block:(i+1)*trials_per_block] = trials[i*trials_per_block:(i+1)*trials_per_block,0] != habit_seq


    trial_type = trial_type.astype('int32')

    fname = 'planning_config_'+'degradation_'+ str(int(contingency_degradation))+ '_switch_' + str(int(switch_cues))\
             + '_train' + str(training_blocks) + '_degr' + str(degradation_blocks) + '_n' + str(trials_per_block)\
             + '_nr_' + str(nr) +'.json'

    # fname = 'test.json'

    if shuffle and blocked:
        subfolder = '/shuffled_and_blocked'
    elif shuffle and not blocked:
        subfolder ='/shuffled'
    elif not shuffle and blocked:
        subfolder ='/blocked'
    elif not shuffle and not blocked:
        subfolder ='/original'

    path = os.path.join(os.getcwd(),'config'+subfolder)
    fname = os.path.join(path, fname)


    if export:
        config = {
                  'context' : context.astype('int32').tolist(),
                  'sequence': trials[:,0].astype('int32').tolist(),
                  'starts': trials[:,1].astype('int32').tolist(),
                  'planets': trials[:,2:-1].astype('int32').tolist(),
                  'exp_reward': trials[:,-1].tolist(),
                  'trial_type': trial_type.tolist(),
                  'block': blocks.astype('int32').tolist(),
                  'degradation_blocks': degradation_blocks,
                  'training_blocks': training_blocks,
                  'interlace': interlace,
                  'contingency_degradation': contingency_degradation,
                  'switch_cues': switch_cues,
                  'trials_per_block': trials_per_block,
                  'blocked': blocked,
                  'shuffle': shuffle, 
                  'miniblock_size' : block,
                  'seed':seed,
                  'nr': nr,
                  'state_transition_matrix':stm,
                }

        with open(fname, "w") as file:
            js.dump(config, file)


def create_config_files_planning(training_blocks, degradation_blocks, extinction_blocks, trials_per_block, habit_seq = 3,\
    shuffle=False, blocked=False, block=None, trials=None, nr=3):

    if trials is None:  

        trials, state_transition_matrix = all_possible_trials(habit_seq=habit_seq,shuffle=shuffle,nr=nr)

    degradation = [True]
    cue_switch = [False]
    arrays = [degradation, cue_switch, degradation_blocks, training_blocks, extinction_blocks, trials_per_block,[blocked]]

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:
        print(l)
        create_trials_planning(trials, contingency_degradation=l[0],switch_cues=l[1],degradation_blocks=l[2],
                            training_blocks=l[3], extinction_blocks=l[4], trials_per_block=l[5], habit_seq = habit_seq,\
                            blocked = l[-1],shuffle=shuffle,block=block,nr=nr,stm=state_transition_matrix)


#%% CALLS
conf = ['shuffled', 'shuffled_and_blocked', 'blocked','original']
data_folder='config'


for con in conf:
    path = os.path.join(data_folder, con)

    if not os.path.exists(path):
        os.makedirs(path)


combinations = []
# create_config_files_planning([4],[2],[42],shuffle=True,blocked=True,block=5)
# create_config_files_planning([4],[6],[70],shuffle=True,blocked=False)
# create_config_files_planning([4],[2],[42],shuffle=True,blocked=True, block=3)
# create_config_files_planning([4],[6],[42],shuffle=True,blocked=True, block=3)
# create_config_files_planning([6],[6],[42],shuffle=True,blocked=True, block=3,nr=2)
# create_config_files_planning([6],[6],[70],shuffle=True,blocked=True, block=5,nr=2)

create_config_files_planning([2],[0],[0],[28],shuffle=True,blocked=True, block=2,nr=3)
create_config_files_planning([1],[1],[1],[28],shuffle=True,blocked=True, block=2,nr=3)

# create_config_files_planning([6],[6],[2],[42],shuffle=True,blocked=True, block=3,nr=3)
# create_config_files_planning([6],[6],[2],[70],shuffle=True,blocked=True, block=5,nr=3)




#%% Average reward in each phase
import json 
import pandas as pd
import numpy as np
import sys

fname = 'config/shuffled_and_blocked/planning_config_degradation_1_switch_0_train6_degr6_n42.json'
fname = 'config/shuffled_and_blocked/planning_config_degradation_1_switch_0_train2_degr0_n42_nr_3.json'

if sys.platform == 'win32':
   fname = fname.replace('/', '\\') 
f = open(fname)

data = json.load(f)
df = pd.DataFrame.from_dict(data)

exp_rewards = df.groupby('block').mean('exp_reward')['exp_reward']
print(exp_rewards[:6].sum()/6)
print(exp_rewards[6:12].sum()/6)
# df.head(10)



#%% OLD FUNCTIONS

def create_trials_two(data, 
                      switch_cues= False,\
                      h1 = 3, h2 = 6,
                      contingency_degradation = True,\
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
    trials = np.zeros([nblocks*trials_per_block, ncols])
    trial_type = np.zeros(nblocks*trials_per_block)
    context = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)

    tt=0

    split_data = [[None for c in range(2)] for s in sequences]

    # I am takign this one as a reference since this occurs in all of them
    # sequence 4 and 6 have actually combinations with expected rewards of -1.85 and -0.95
    unique = np.unique(data[0][0][:,-1])
    probs = (np.arange(unique.size)+1)/(np.arange(unique.size)+1).sum()
    rewards = np.random.choice(np.arange(unique.size), p=probs, size=ntrials)


    for s in range(len(data)):
        for c in range(len(data[s])):
            print(unique)
            dat = data[s][c]
            split_data[s][c] = [dat[dat[:,-1]==k] for k in unique]
            for k in range(unique.size):
                np.random.shuffle(split_data[s][c][k])
    
    if block is not None:
        miniblocks = half_block//block

    for i in range(nblocks):

        if i >= training_blocks and i < training_blocks + degradation_blocks:
            contingency = 1
            tt = 1
        else: 
            contingency = 0
            tt = 0 

        if i >= (nblocks - extinction_blocks): 
            tt = 2

        if not blocked:

            # populate habit seq trials
            for t in range(half_block):
                ri = half_block*i + t

                tr1 = int(i*trials_per_block + t)
                trials[tr1,:] = split_data[h1][contingency][rewards[ri]][0]
                split_data[h1][contingency][rewards[ri]] = \
                    np.roll(split_data[h1][contingency][rewards[ri]], -1, axis=0)

                tr2 = int(i*trials_per_block + half_block + t)
                trials[tr2,:] = split_data[h2][contingency][rewards[ri]][0]
                split_data[h2][contingency][rewards[ri]] = \
                    np.roll(split_data[h2][contingency][rewards[ri]], -1, axis=0)

            np.random.shuffle(trials[i*trials_per_block:(i+1)*trials_per_block,:])

        if blocked:
            
            # extract all planning trials and shuffle them
            h1_trials = np.zeros([half_block, trials.shape[1]])
            h2_trials = np.zeros([half_block, trials.shape[1]])
            
            for t in range(half_block):
                tr = int(i*trials_per_block + half_block + t)
                ri = half_block*i + t

                h1_trials[t,:] = split_data[h1][contingency][rewards[ri]][0]
                split_data[h1][contingency][rewards[ri]] = \
                    np.roll(split_data[h1][contingency][rewards[ri]], -1, axis=0)
            

                h2_trials[t,:] = split_data[h2][contingency][rewards[ri]][0]
                split_data[h2][contingency][rewards[ri]] = \
                    np.roll(split_data[h2][contingency][rewards[ri]], -1, axis=0)

            for mb in range(miniblocks):
                tr = int(i*trials_per_block)
                trials[tr+2*mb*block:tr+(2*mb+1)*block] = h1_trials[mb*block:(mb+1)*block,:]
                trials[tr+(2*mb+1)*block:tr+(2*mb+2)*block] = h2_trials[mb*block:(mb+1)*block,:]
            
        trial_type[i*trials_per_block:(i+1)*trials_per_block] = int(tt)
        blocks[i*trials_per_block:(i+1)*trials_per_block] = int(i)
        context[i*trials_per_block:(i+1)*trials_per_block] = trials[i*trials_per_block:(i+1)*trials_per_block,0] != h1


    trial_type = trial_type.astype('int32')

    fname = 'config_'+'degradation_'+ str(int(contingency_degradation))+ '_switch_' + str(int(switch_cues))\
             + '_train' + str(training_blocks) + '_degr' + str(degradation_blocks) + '_n' + str(trials_per_block)+'.json'

    if shuffle and blocked:
        subfolder = '/shuffled_and_blocked'
    elif shuffle and not blocked:
        subfolder ='/shuffled'
    elif not shuffle and blocked:
        subfolder ='/blocked'
    elif not shuffle and not blocked:
        subfolder ='/original'

    path = os.path.join(os.getcwd(),'config'+subfolder)
    fname = os.path.join(path, fname)



    if export:
        config = {
                  'context' : context.astype('int32').tolist(),
                  'sequence': trials[:,0].astype('int32').tolist(),
                  'starts': trials[:,1].astype('int32').tolist(),
                  'planets': trials[:,2:-1].astype('int32').tolist(),
                  'exp_reward': trials[:,-1].tolist(),
                  'trial_type': trial_type.tolist(),
                  'block': blocks.astype('int32').tolist(),
                  'degradation_blocks': degradation_blocks,
                  'training_blocks': training_blocks,
                  'interlace': interlace,
                  'contingency_degradation': contingency_degradation,
                  'switch_cues': switch_cues,
                  'trials_per_block': trials_per_block,
                  'blocked': blocked,
                  'shuffle': shuffle, 
                  'miniblock_size' : block,
                  'seed':seed
                }

        with open(fname, "w") as file:
            js.dump(config, file)


def create_config_files(training_blocks, degradation_blocks, trials_per_block,\
                        trials=None, shuffle=False, blocked=False, block=None):

    if trials is None:
        trials = all_possible_trials(habit_seq=3, shuffle=shuffle)
    
    degradation = [True]
    cue_switch = [False]

    arrays = [degradation, cue_switch, degradation_blocks, training_blocks, trials_per_block]

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:

        create_trials_two(trials, contingency_degradation=l[0],switch_cues=l[1],\
            degradation_blocks=l[2], training_blocks=l[3], trials_per_block=l[4],\
            shuffle=shuffle, blocked=blocked,block=block)

