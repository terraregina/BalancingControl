#%%                     
import json  as js
import numpy as np
import os
from itertools import product, repeat
import pickle
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
from planet_sequences import generate_trials_df


def all_possible_trials_planning(habit_seq=3):
    
    ns=6
    all_rewards = [[-1,1,1], [1,1,-1]]
    sequences  = np.arange(8)
    slices = [None]*2
    planet_confs = [None]*2
    for ri, rewards in enumerate(all_rewards):
        slices[ri], planet_confs[ri] = generate_trials_df(rewards, sequences)


    data = [[] for i in range(len(sequences))]

    for s in sequences:
        for ci, trials_for_given_contingency in enumerate(slices):
            slice = trials_for_given_contingency[s]
            planet_conf = planet_confs[ci]
            ntrials = slice.shape[0]
            dt = np.zeros([slice.shape[0], 2 + ns])
            plnts = slice.planet_conf                          # configuration indeces for s1 trials
            plnts = planet_conf[[plnts.values.tolist()]].tolist() # actual planet configurations for s1 trials
            strts = slice.start.values.tolist()                # starting points for s1 trials 

            # dt[:,0] = [0]*ntrials      # context index
            dt[:,0] = [s]*ntrials     # optimal sequence index
            dt[:,1] = strts       # trial starting position
            dt[:,2:] = plnts      # planets 
            data[s].append(dt)



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

    return data_extended


def create_trials_planning(data, habit_seq = 3, contingency_degradation = True,\
                        switch_cues= False,\
                        training_blocks = 2,\
                        degradation_blocks = 1,\
                        interlace = True,\
                        trials_per_block = 28, export = True):

    sequences = np.arange(8)
    np.random.seed(1)
    if trials_per_block % 2 != 0:
        raise Exception('Give even number of trials per block!')


    fname = 'planning_config_'+'degradation_'+ str(int(contingency_degradation))+ '_switch_' + str(int(switch_cues))\
             + '_train' + str(training_blocks) + '_degr' + str(degradation_blocks) + '_n' + str(trials_per_block)+'.json'


    nblocks = training_blocks + degradation_blocks + 2
    half_block = np.int(trials_per_block/2)
    
    ncols = data[0][0].shape[1]
    # block = np.int(trials[0].shape[0]/nblocks)
    trials = np.zeros([nblocks*trials_per_block, ncols])
    trial_type = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)
    n_planning_seqs = sequences.size -1
    planning_seqs = sequences[sequences != habit_seq]
    shift = int(half_block/n_planning_seqs)
    tt=0
    # populate training blocks and extinction block
    for i in range(nblocks):
        if i == (nblocks -2):                   # if now populating extinction block
            tt = 2

        trials[(2*i)*half_block : (2*i+1)*half_block , :] = data[habit_seq][0][i*half_block : (i+1)*half_block,:]
        planning_trials = np.zeros([half_block, ncols])
        for si, s in enumerate(planning_seqs):
            planning_trials[si*shift:(si+1)*shift,:] = data[s][0][i*shift:(i+1)*shift,:]           
        trials[(2*i+1)*half_block : (2*i+2)*half_block , :] = planning_trials

        trial_type[(2*i)*half_block:(2*i+2)*half_block] = tt
        blocks[(2*i)*half_block:(2*i+2)*half_block] = i
        

    # populate the degradation blocks
    for i in range(training_blocks, degradation_blocks + training_blocks):
        trial_type[(2*i)*half_block:(2*i+2)*half_block] = 1

        if contingency_degradation:
            ind = 1
        else:
            ind = 0

        trials[(2*i)*half_block : (2*i+1)*half_block , :] = data[habit_seq][ind][i*half_block:(i+1)*half_block,:]
        planning_trials = np.zeros([half_block, ncols])
        for si, s in enumerate(planning_seqs):
            planning_trials[si*shift:(si+1)*shift,:] = data[s][ind][i*shift:(i+1)*shift,:]
        trials[(2*i+1)*half_block : (2*i+2)*half_block , :] = planning_trials

        if switch_cues:
            trials[:,0] = trials[:,0] == 0

    if interlace:
        for i in range(nblocks):
            np.random.shuffle(trials[(2*i)*half_block:(2*i+2)*half_block,:])
            
    trials = trials.astype('int32')
    trial_type = trial_type.astype('int32')

    path = os.path.join(os.getcwd(),'config')
    fname = os.path.join(path, fname)
    if export:
        config = {'context' : trials[:,0].tolist(),
                  'sequence': trials[:,1].tolist(),
                  'starts': trials[:,2].tolist(),
                  'planets': trials[:,3:].tolist(),
                  'trial_type': trial_type.tolist(),
                  'block': blocks.tolist(),
                  'degradation_blocks': degradation_blocks,
                  'training_blocks': training_blocks,
                  'interlace': interlace,
                  'contingency_degradation': contingency_degradation,
                  'switch_cues': switch_cues,
                  'trials_per_block': trials_per_block
                }

        with open(fname, "w") as file:
            js.dump(config, file)



def create_config_files_planning(training_blocks, degradation_blocks, trials_per_block, habit_seq = 3):
    trials = all_possible_trials_planning(habit_seq=habit_seq)
    degradation = [True]
    cue_switch = [False]
    arrays = [degradation, cue_switch, degradation_blocks, training_blocks, trials_per_block]

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:
        print(l)
        create_trials_planning(trials, contingency_degradation=l[0],switch_cues=l[1],degradation_blocks=l[2],
                            training_blocks=l[3], trials_per_block=l[4], habit_seq = habit_seq)


#%%
def all_possible_trials_two_seqs():
    

    ns=6
    all_rewards = [[-1,1,1], [1,1,-1]]
    trials = []

    for rewards in all_rewards:
        s1 = 3       # optimal sequence 1
        s2 = 6       # optimal sequence 2

        slices, planet_confs = generate_trials_df(rewards, [s1,s2])
        # slices == data for s1 and s2
        # planet_conf == all possible planet configurations overall
        
        n1 = slices[0].shape[0]      # number of trials for s1
        n2 = slices[1].shape[0]      # number of trials for s2

        dt = np.zeros([n1 + n2, 3 + ns])  # data storage array

        plnts = slices[0].planet_conf                          # configuration indeces for s1 trials
        plnts = planet_confs[[plnts.values.tolist()]].tolist() # actual planet configurations for s1 trials
        strts = slices[0].start.values.tolist()                # starting points for s1 trials 

        dt[0:n1,0] = [0]*n1      # context index
        dt[0:n1,1] = [s1]*n1     # optimal sequence index
        dt[0:n1,2] = strts       # trial starting position
        dt[0:n1,3:] = plnts      # planets 

        # repeat the same thing for s2
        plnts = slices[1].planet_conf
        plnts = planet_confs[[plnts.values.tolist()]].tolist()
        strts = slices[1].start.values.tolist()

        dt[n1:n1+n2,0] = [1]*n2
        dt[n1:n1+n2,1] = [s2]*n2
        dt[n1:n1+n2,2] = strts
        dt[n1:n1+n2,3:] = plnts
        trials.append(dt)

    return trials



'''
function that actually creates the trials for a given experimental version
'''

def create_trials_two_seqs(trials_orig, export=True,
                           contingency_degradation = False,
                           switch_cues = True,
                           training_blocks=4,
                           degradation_blocks = 2,
                           trials_per_block=60, interlace = True):
 
    np.random.seed(1)
    if trials_per_block % 2 != 0:
        raise Exception('Give even number of trials per block!')


    fname = 'config_'+'degradation_'+ str(int(contingency_degradation))+ '_switch_' + str(int(switch_cues))\
             + '_train' + str(training_blocks) + '_degr' + str(degradation_blocks) + '_n' + str(trials_per_block)+'.json'

    start = np.argmax(trials_orig[0][:,1])
    inds = np.concatenate((np.arange(start), np.arange(start*2, start*3,1), np.arange(start,start*2,1), np.arange(start*3,start*4,1)))
    trials = [None]*len(trials_orig)
    for ti, tr in enumerate(trials_orig):            
        trials[ti] = np.tile(tr[:start*2,:], (2,1))
        trials[ti] = trials[ti][inds,:]

    nblocks = training_blocks + degradation_blocks + 2
    shift = np.int(trials[0].shape[0]/2)
    half_block = np.int(trials_per_block/2)
    

    # block = np.int(trials[0].shape[0]/nblocks)
    data = np.zeros([nblocks*trials_per_block, trials[0].shape[1]])
    trial_type = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)
    tt=0

    # populate training blocks and extinction block
    for i in range(nblocks):
        if i >= training_blocks:                   # if now populating extinction block
            # i = training_blocks + degradation_blocks
            tt = 2

        data[(2*i)*half_block : (2*i+1)*half_block , :] = trials[0][i*half_block : (i+1)*half_block,:]
        data[(2*i+1)*half_block : (2*i+2)*half_block , :] = trials[0][shift+(i)*half_block : shift+(i+1)*half_block,:]

        trial_type[(2*i)*half_block:(2*i+2)*half_block] = tt
        blocks[(2*i)*half_block:(2*i+2)*half_block] = i


    # populate the degradation blocks
    for i in range(training_blocks, degradation_blocks + training_blocks):
        blocks[(2*i)*half_block:(2*i+2)*half_block] = i
        trial_type[(2*i)*half_block:(2*i+2)*half_block] = 1

        if contingency_degradation:
            ind = 1
        else:
            ind = 0

        data[(2*i)*half_block : (2*i+1)*half_block , :] = trials[ind][i*half_block : (i+1)*half_block,:]
        data[(2*i+1)*half_block : (2*i+2)*half_block , :] = trials[ind][shift+(i)*half_block : shift+(i+1)*half_block,:]
        # np.random.shuffle(data[(2*i)*half_block:(2*i+2)*half_block,:])

        if switch_cues:
            data[(2*i)*half_block:(2*i+2)*half_block,0] = data[(2*i)*half_block:(2*i+2)*half_block,0] == 0

    if interlace:
        for i in range(nblocks):
            np.random.shuffle(data[(2*i)*half_block:(2*i+2)*half_block,:])
            
    data = data.astype('int32')
    trial_type = trial_type.astype('int32')


    path = os.path.join(os.getcwd(),'config')
    fname = os.path.join(path, fname)
    
    if export:
        config = {'context' : data[:,0].tolist(),
                  'sequence': data[:,1].tolist(),
                  'starts': data[:,2].tolist(),
                  'planets': data[:,3:].tolist(),
                  'trial_type': trial_type.tolist(),
                  'block': blocks.tolist(),
                  'degradation_blocks': degradation_blocks,
                  'training_blocks': training_blocks,
                  'interlace': interlace,
                  'contingency_degradation': contingency_degradation,
                  'switch_cues': switch_cues,
                  'trials_per_block': trials_per_block
                }

        with open(fname, "w") as file:
            js.dump(config, file)


'''
wrapper function which creates all config files containing information about planet configuration,
start position and context cue which are loaded 
in by the run_single_sim function to simulate an agent

'''
def create_config_files(training_blocks, degradation_blocks, trials_per_block):
    trials = all_possible_trials_two_seqs()
    degradation = [True]
    cue_switch = [False]
    arrays = [degradation, cue_switch, degradation_blocks, training_blocks, trials_per_block]

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:
        print(l)

        create_trials_two_seqs(trials, contingency_degradation=l[0],switch_cues=l[1],degradation_blocks=l[2],
                            training_blocks=l[3], trials_per_block=l[4])


def create_trials_two_seqs_sequence_dependent_degradation(trials_orig, export=False,
                           contingency_degradation = False,
                           switch_cues = True,
                           training_blocks=4,
                           degradation_blocks = 2,
                           trials_per_block=60, interlace = True):
 
    np.random.seed(1)
    if trials_per_block % 2 != 0:
        raise Exception('Give even number of trials per block!')


    fname = 'differential_config_'+'degradation_'+ str(int(contingency_degradation))+ '_switch_' + str(int(switch_cues))\
             + '_train' + str(training_blocks) + '_degr' + str(degradation_blocks) + '_n' + str(trials_per_block)+'.json'

    start = np.argmax(trials_orig[0][:,1])
    # inds = np.concatenate((np.arange(start), np.arange(start*2, start*3,1), np.arange(start,start*2,1), np.arange(start*3,start*4,1)))
    trials = [None]*len(trials_orig)
    rep = 5
    for ti, tr in enumerate(trials_orig):
        seq1 = trials_orig[ti][:start,:]
        seq2 = trials_orig[ti][start:start*2,:]
        trials[ti] = np.concatenate((np.tile(seq1,(rep,1)), np.tile(seq2,(rep,1))))            
        # # trials[ti] = np.tile(tr[:start*2,:], (2,1))
        # # trials[ti] = trials[ti][inds,:]
        # trials[ti] = tr[:start*2,:]



    nblocks = training_blocks + degradation_blocks + 2
    shift = np.int(trials[0].shape[0]/2)
    half_block = np.int(trials_per_block/2)
    

    # block = np.int(trials[0].shape[0]/nblocks)
    data = np.zeros([nblocks*trials_per_block, trials[0].shape[1]])
    trial_type = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)
    tt=0
    degradation_context = 0

    # populate training blocks and extinction block
    for i in range(nblocks):
        if i >= training_blocks + degradation_blocks:
            tt = 2

        data[(2*i)*half_block : (2*i+1)*half_block , :] = trials[0][i*half_block : (i+1)*half_block,:]
        data[(2*i+1)*half_block : (2*i+2)*half_block , :] = trials[0][shift+(i)*half_block : shift+(i+1)*half_block,:]

        trial_type[(2*i)*half_block:(2*i+2)*half_block] = tt
        blocks[(2*i)*half_block:(2*i+2)*half_block] = i
        

    # populate the degradation blocks
    for i in range(training_blocks, degradation_blocks + training_blocks):
        trial_type[(2*i)*half_block:(2*i+2)*half_block] = 1

        data[(2*i)*half_block : (2*i+1)*half_block , :] = trials[1][i*half_block : (i+1)*half_block,:]
        # data[(2*i+1)*half_block : (2*i+2)*half_block , :] = trials[0][shift+(i)*half_block : shift+(i+1)*half_block,:]
        # np.random.shuffle(data[(2*i)*half_block:(2*i+2)*half_block,:])

        # if switch_cues:
        #     data[(2*i)*half_block:(2*i+2)*half_block,0] = data[(2*i)*half_block:(2*i+2)*half_block,0] == 0

    if interlace:
        for i in range(nblocks):
            np.random.shuffle(data[(2*i)*half_block:(2*i+2)*half_block,:])
            
    data = data.astype('int32')
    trial_type = trial_type.astype('int32')

    path = os.path.join(os.getcwd(),'config')
    fname = os.path.join(path, fname)
    if export:
        config = {'context' : data[:,0].tolist(),
                  'sequence': data[:,1].tolist(),
                  'starts': data[:,2].tolist(),
                  'planets': data[:,3:].tolist(),
                  'trial_type': trial_type.tolist(),
                  'block': blocks.tolist(),
                  'degradation_blocks': degradation_blocks,
                  'training_blocks': training_blocks,
                  'interlace': interlace,
                  'contingency_degradation': contingency_degradation,
                  'switch_cues': switch_cues,
                  'trials_per_block': trials_per_block,
                  'degradation_context': degradation_context
                }

        with open(fname, "w") as file:
            js.dump(config, file)

def create_config_files_context_dependent_degradation(training_blocks, degradation_blocks, trials_per_block):
    trials = all_possible_trials_two_seqs()
    degradation = [True]
    cue_switch = [False]
    arrays = [degradation, cue_switch, degradation_blocks, training_blocks, trials_per_block]

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:
        print(l)

        create_trials_two_seqs_sequence_dependent_degradation(trials, contingency_degradation=l[0],switch_cues=l[1],degradation_blocks=l[2],
                            training_blocks=l[3], trials_per_block=l[4],export=True)
 
#%%

# create_config_files_planning([4],[2,4,6],[70])
create_config_files([2], [2], [40])
# create_config_files_context_dependent_degradation([4],[2,4,6],[70])

# %%

# fname = 'config/' + 'config_degradation_1_switch_0_train4_degr2_n70.json' 
# file = os.path.join(os.getcwd(), fname)
# import pandas as pd
# file = open(file, 'r')
# data = js.load(file)
# df = pd.DataFrame(data)
# df.query('trial_type == 2').tail(30)

