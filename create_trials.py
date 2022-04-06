                               
import json  as js
import numpy as np
import os
from itertools import product, repeat
import pickle
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
from planet_sequences import generate_trials_df

def all_possible_trials():
    

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

    nblocks = training_blocks + degradation_blocks + 1
    shift = np.int(trials[0].shape[0]/2)
    half_block = np.int(trials_per_block/2)
    

    # block = np.int(trials[0].shape[0]/nblocks)
    data = np.zeros([nblocks*trials_per_block, trials[0].shape[1]])
    trial_type = np.zeros(nblocks*trials_per_block)
    blocks = np.zeros(nblocks*trials_per_block)
    tt=0

    # populate training blocks and extinction block
    for i in range(training_blocks+1):
        if i == training_blocks:                   # if now populating extinction block
            i = training_blocks + degradation_blocks
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
        for i in range(degradation_blocks + training_blocks+1):
            np.random.shuffle(data[(2*i)*half_block:(2*i+2)*half_block,:])
            
    data = data.astype('int32')
    trial_type = trial_type.astype('int32')

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
    trials = all_possible_trials()
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
        for i in range(degradation_blocks + training_blocks+1):
            np.random.shuffle(data[(2*i)*half_block:(2*i+2)*half_block,:])
            
    data = data.astype('int32')
    trial_type = trial_type.astype('int32')

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
    trials = all_possible_trials()
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

# (trials_orig, export=False,
#                            contingency_degradation = False,
#                            switch_cues = True,
#                            training_blocks=4,
#                            degradation_blocks = 2,
#                            trials_per_block=60, interlace = True):
 


# old configs where no differential degradation
# create_config_files([4], [2,4,6], [60])

# files with differential extinction

create_config_files_context_dependent_degradation([4],[4,6],[60])