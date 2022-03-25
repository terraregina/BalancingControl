### #     # ######  ####### ######  #######  #####  
 #  ##   ## #     # #     # #     #    #    #     # 
 #  # # # # #     # #     # #     #    #    #       
 #  #  #  # ######  #     # ######     #     #####  
 #  #     # #       #     # #   #      #          # 
 #  #     # #       #     # #    #     #    #     # 
### #     # #       ####### #     #    #     #####  
                                                    
#%%                                          
import json  as js
from venv import create
from matplotlib.style import context
import numpy as np
import itertools
import os
import action_selection as asl
import seaborn as sns
from itertools import product
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import concurrent.futures
import time

import perception as prc
import agent as agt
from environment import PlanetWorld
from agent import BayesianPlanner
from world import World
from planet_sequences import generate_trials_df


#%%

####### #     # #     #  #####  ####### ### ####### #     #       ######  ####### #######  #####  
#       #     # ##    # #     #    #     #  #     # ##    #       #     # #       #       #     # 
#       #     # # #   # #          #     #  #     # # #   #       #     # #       #       #       
#####   #     # #  #  # #          #     #  #     # #  #  #       #     # #####   #####    #####  
#       #     # #   # # #          #     #  #     # #   # #       #     # #       #             # 
#       #     # #    ## #     #    #     #  #     # #    ##       #     # #       #       #     # 
#        #####  #     #  #####     #    ### ####### #     #       ######  ####### #        #####  
                                                                                               



def run_agent(par_list, trials, T, ns=6, na=2, nr=3, nc=2, npl=2):

    # learn_pol          = initial concentration parameters for POLICY PRIOR
    # context_trans_prob = probability of staing in a given context, int
    # avg                = average vs maximum selection, True for avg
    # Rho                = environment reward generation probability as a function of time, dims: trials x nr x ns
    # utility            = GOAL PRIOR, preference p(o)
    # B                  = state transition matrix depedenent on action: ns x ns x actions
    # npl                = number of unique planets accounted for in the reward contingency representation  
    # C_beta           = phi or estimate of p(reward|planet,context) 
    
    learn_pol, context_trans_prob, cue_ambiguity, avg, Rho, utility, B, planets, starts, colors, rc, learn_rew = par_list


    """
    create matrices
    """

    #generating probability of observations in each state
    A = np.eye(ns)


    # agent's initial estimate of reward generation probability
    C_beta = rc.copy()
 
    C_agent = np.zeros(C_beta.shape)        # nr x npl; default order is r=(1, 0, 1) and s=(0,1,2) 

    for c in range(nc):
        C_agent[:,:,c] = np.array([(C_beta[:,i,c])/(C_beta[:,i,c]).sum() for i in range(npl)]).T


    """
    initialize context transition matrix
    """

    p = context_trans_prob
    q = (1-p)/(nc-1)

    transition_matrix_context = np.zeros([nc,nc]) + q
    transition_matrix_context = transition_matrix_context - np.eye(nc)*q + np.eye(nc)*p 



    """ 
    create environment class
    """
    
    environment = PlanetWorld(A,
                              B,
                              Rho,
                              planets,
                              starts,
                              colors,
                              trials,
                              T,
                              ns,
                              npl,
                              nr,
                              na)


    """ 
    create policies and setup concentration parameters
    The pseudo counts alpha^t_ln which parameterize the prior over actions for
    """

    pols = np.array(list(itertools.product(list(range(na)), repeat=T-1)))
    npi = pols.shape[0]


    C_alphas = np.zeros((npi, nc)) + learn_pol
    prior_pi = C_alphas / C_alphas.sum(axis=0)

    """
    set state prior
    """

    state_prior = np.ones((ns))
    state_prior = state_prior/ state_prior.sum() 


    """
    set action selection method
    """

    if avg:

        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)
    else:

        ac_sel = asl.MaxSelector(trials = trials, T = T,
                                      number_of_actions = na)

    """
    set context prior
    """

    prior_context = np.zeros((nc)) + 0.1/(nc-1)
    prior_context[0] = 0.9

    """
    define generative model for context
    """


    no =  np.unique(colors).size                    # number of observations (background color)
    C = np.zeros([no, nc])
    p = cue_ambiguity                               # how strongly agent associates context observation with a particular context       
    dp = 0.001
    p2 = 1 - p - dp
    C[0,:] = [p,dp/2,p2,dp/2]
    C[1,:] = [dp/2, p, dp/2, p2]


    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, 
                                           B, 
                                           C_agent, 
                                           transition_matrix_context, 
                                           state_prior, 
                                           utility, 
                                           prior_pi, 
                                           C_alphas,
                                           C_beta,
                                           generative_model_context = C,
                                           T=T)


    # agent
    bayes_pln = agt.BayesianPlanner(bayes_prc,
                                    ac_sel,
                                    pols,
                                    prior_states = state_prior,
                                    prior_policies = prior_pi,
                                    trials = trials,
                                    prior_context = prior_context,
                                    learn_habit=True,
                                    learn_rew = learn_rew,
                                    number_of_planets = npl
                                    )


    """
    create world
    """

    w = World(environment, bayes_pln, trials = trials, T = T)
    bayes_pln.world = w

    """
    simulate experiment
    """

    w.simulate_experiment(range(trials))
    w.h = learn_pol
    w.q = context_trans_prob
    w.p = cue_ambiguity

    return w



'''
function creating all possible trials and starting points
for given optimal sequence (s1 and s2) and planet reward association (all_rewards)

if you want reverse contingenvies you need to give [[1,1,-1]], this will produce
the following p(reward|planet)
[0,  , 0   , 0.95]
[0.05, 0.95, 0.05]
[0.95, 0.05, 0   ]
where rows are possible rewards [-1,0,1] and cols possible planet types [red, gray, green] for example

OUTPUT: list with all trial information for all possible trials for a given optimal sequence pair
'''
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
wrapper function which creates all config files containing information about planet configuration,
start position and context cue which are loaded 
in by the run_single_sim function to simulate an agent

'''
def create_config_files(training_blocks, degradation_blocks, trials_per_block):
    trials = all_possible_trials()
    degradation = [True, False]
    cue_switch = [True, False]
    arrays = [cue_switch, degradation, degradation_blocks, training_blocks, trials_per_block]
    # function will bug out if training_blocks*trials_per_block > trials[0].shape[0]
    # aka if trials are done more than once. this is about 530 trials I think
    # I can also repeat the trials, was just a bit lazy

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    for l in lst:
        print(l)
        create_trials_two_seqs(trials, contingency_degradation=l[0],switch_cues=l[1],degradation_blocks=l[2],
                               training_blocks=l[3], trials_per_block=l[4])


'''
function that actually creates the trials for a given experimental version
'''

def create_trials_two_seqs(trials, export=True,
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
                  'block': blocks.tolist()
                }

        with open(fname, "w") as file:
            js.dump(config, file)


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
                                repetitions,
                                fname, reward_naive=True):

    folder = os.getcwd()
    switch_cues, contingency_degradation, learn_rew, context_trans_prob, cue_ambiguity, h  = lst

    file = open(os.path.join(folder,fname))

    task_params = js.load(file)                                                                                 

    colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
    sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
    starts = np.asarray(task_params['starts'])           # starting position of agent
    planets = np.asarray(task_params['planets'])         # planet positions 
    trial_type = np.asarray(task_params['trial_type'])
    blocks = np.asarray(task_params['block'])


    nblocks = int(blocks.max()+1)         # number of blocks
    trials = blocks.size                  # number of trials
    block = int(trials/nblocks)           # trials per block
 
    meta = {
        'trial_file' : fname, 
        'trial_type' : trial_type,
        'switch_cues': switch_cues == True,
        'contingency_degradation' : contingency_degradation == True,
        'learn_rew' : learn_rew == True,
        'context_trans_prob': context_trans_prob,
        'cue_ambiguity' : cue_ambiguity,
        'h' : h,
        'optimal_sequence' : sequence,
        'blocks' : blocks,
        'trials' : trials,
        'nblocks' : nblocks,
        'trials_per_block': block,
    }

    all_optimal_seqs = np.unique(sequence)                                                                            

    # reward probabilities schedule dependent on trial and planet constelation
    Rho = np.zeros([trials, nr, ns])

    for i, pl in enumerate(planets):
        if i >= block*(nblocks-2) and i < block*(nblocks-1) and contingency_degradation:
            # print(i)
            # print(pl)
            Rho[i,:,:] = planet_reward_probs_switched[[pl]].T
        else:
            Rho[i,:,:] = planet_reward_probs[[pl]].T

    u = 0.99
    utility = np.array([(1-u)/2,(1-u)/2,u])

    if reward_naive==True:
        reward_counts = np.ones([nr, npl, nc])
    else:
        reward_counts = np.tile(planet_reward_probs.T[:,:,np.newaxis]*100,(1,1,nc))

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
                learn_rew]

    prefix = ''
    if switch_cues == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if contingency_degradation == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'

    worlds = [run_agent(par_list, trials, T, ns , na, nr, nc, npl) for _ in range(repetitions)]

    worlds.append(meta)
    fname = prefix +'p' + str(cue_ambiguity) + '_q' + str(context_trans_prob) + '_h' + str(h) + '.json'
    fname = os.path.join(os.path.join(folder,'data'), fname)
    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(worlds)
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)

    return fname



""""""
def main():

    na = 2                                           # number of unique possible actions
    nc = 4                                           # number of contexts, planning and habit
    nr = 3                                           # number of rewards
    ns = 6                                           # number of unique travel locations
    npl = 3
    steps = 3                                        # numbe of decisions made in an episode
    T = steps + 1                                    # episode length

    # reward probabiltiy vector
    repetitions = 1
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

    # setup simulation parameters here
    h =  [1,100]
    cue_ambiguity = [0.8]                      # cue ambiguity reffers to how certain agent is a given observation refers to a given context
    context_trans_prob = [nc]                  # the higher the value the more peaked the distribution is
    degradation = [True]                       # bit counter intuitive, should change the name :D
    cue_switch = [True]
    learn_rew = [True]
    arrays = [cue_switch, degradation, learn_rew, context_trans_prob, cue_ambiguity,h]
    training_blocks = [4]
    degradation_blocks=[2]
    trials_per_block=[60]

    data_path = os.path.join(os.getcwd(),'data')
    if not os.path.isdir(data_path): 
        os.mkdir(data_path)
    # needs to be run only the first time you use main
    create_config_files(training_blocks, degradation_blocks, trials_per_block)

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    # failed efforts at parallel running of sims
    # constant_arguments = [ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs, planet_reward_probs_switched,repetitions]
    # n = len(lst)
    # ca = [None]*len(constant_arguments)
    # for i,arg in enumerate(constant_arguments):
    #     ca[i] = [arg]*n


    # start = time.perf_counter()

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = executor.map(run_single_sim, lst, ca[0],ca[1],ca[2],ca[3],ca[4],ca[5],ca[6],ca[7],ca[8],ca[9])
    #     # for result in results:
    #     #     print(result)
    # finish = time.perf_counter()
    # print(f'Finished in {round(finish-start, 2)} second(s) for multithreader')

    start = time.perf_counter()
    db = degradation_blocks[0]
    tb = training_blocks[0]
    tpb = trials_per_block[0]

    for l in lst:
        # name of experiment config file
        config = 'config_'+'degradation_'+ str(int(l[1]))+ '_switch_' + str(int(l[0]))\
                + '_train' + str(tb) + '_degr' + str(db) + '_n' + str(tpb)+'.json'

        run_single_sim(l, ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs, planet_reward_probs_switched,repetitions, config,reward_naive=True)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s) for individual ones')


# ################################################

if __name__ == '__main__':
    main()













# def run_single_sim(lst,repetitions=1,return_name=False):

#     folder = os.getcwd()
#     for l in lst:
#         switch_cues, contingency_degradation, context_trans_prob, cue_ambiguity, h  = l

#         if contingency_degradation and not switch_cues:
#             fname = 'config_degradation_1_switch_0.json'
#         elif contingency_degradation and switch_cues:
#             fname = 'config_degradation_1_switch_1.json'
#         elif not contingency_degradation and not switch_cues:
#             fname = 'config_degradation_0_switch_0.json'
#         elif not contingency_degradation and switch_cues:
#             fname = 'config_degradation_0_switch_1.json'

#         try:
#             file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)
#         except:
#             create_trials(contingency_degradation=contingency_degradation, switch_cues=switch_cues)
#             file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)


#         task_params = js.load(file)                                                                                 


#         np.random.seed(1)
#         colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
#         sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
#         starts = np.asarray(task_params['starts'])           # starting position of agent
#         planets = np.asarray(task_params['planets'])         # planet positions 
#         trial_type = np.asarray(task_params['trial_type'])
#         blocks = np.asarray(task_params['block'])


#         trials = starts.shape[0]                         # number of episodes
#         nblocks = int(blocks.max()+1)
#         trials = blocks.size
#         block = int(trials/nblocks)

#         all_optimal_seqs = np.unique(sequence)                                                                            

#         # define reward probabilities dependent on time and position for reward generation process 
#         Rho = np.zeros([trials, nr, ns])

#         for i, pl in enumerate(planets):
#             if i >= block*(nblocks-2) and i < block*(nblocks-1) and contingency_degradation:
#                 # print(i)
#                 # print(pl)
#                 Rho[i,:,:] = planet_reward_probs_switched[[pl]].T
#                 print(Rho[i,::])
#             else:
#                 Rho[i,:,:] = planet_reward_probs[[pl]].T


#         u = 0.99
#         utility = np.array([(1-u)/2,(1-u)/2,u])




#         reward_counts = np.ones([nr, npl, nc])
#         par_list = [h,                        \
#                     context_trans_prob,       \
#                     cue_ambiguity,            \
#                     'avg',                    \
#                     Rho,                      \
#                     utility,                  \
#                     state_transition_matrix,  \
#                     planets,                  \
#                     starts,                   \
#                     colors,
#                     reward_counts]

#         prefix = ''
#         if switch_cues == True:
#             prefix += 'switch1_'
#         else:
#             prefix +='switch0_'

#         if contingency_degradation == True:
#             prefix += 'degr1_'
#         else:
#             prefix += 'degr0_'

#         for r in range(repetitions):
#             fname = prefix +'p' + str(cue_ambiguity) + '_q' + str(context_trans_prob) + '_h' + str(h) + '_run'+str(r) + '.json'
#             run = run_agent(par_list, trials, T, ns , na, nr, nc, npl) 
#             run.environment.true_optimal = sequence
#             run.environment.trial_type = trial_type
#             fname = os.path.join(folder, fname)
#             print(fname)

#             jsonpickle_numpy.register_handlers()
#             pickled = pickle.encode(run)
#             with open(fname, 'w') as outfile:
#                 json.dump(pickled, outfile)
    
#     if return_name:
#         return fname

# def run_mulitple_sims():
#     h =  [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
#     cue_ambiguity = [ 0.5, 0.6, 0.7, 0.8]
#     context_trans_prob = np.arange(1/nc, 0.7, 0.1)
#     degradation = [True, False]
#     cue_switch = [True, False]
#     repetitions = 5

#     # arrays = [cue_switch, degradation, context_trans_prob, cue_ambiguity,h]
#     # lst = []
#     # for i in product(*arrays):
#     #     lst.append(i)
#     # folder = os.path.join(os.getcwd(),'data') 

#     lst = [[True, False, 1/nc, 0.9, 1]]
#     folder = os.getcwd()
#     for l in lst:
#         switch_cues, contingency_degradation, context_trans_prob, cue_ambiguity, h  = l

#         if contingency_degradation and not switch_cues:
#             fname = 'config_degradation_1_switch_0.json'
#         elif contingency_degradation and switch_cues:
#             fname = 'config_degradation_1_switch_1.json'
#         elif not contingency_degradation and not switch_cues:
#             fname = 'config_degradation_0_switch_0.json'
#         elif not contingency_degradation and switch_cues:
#             fname = 'config_degradation_0_switch_1.json'

#         try:
#             file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)
#         except:
#             create_trials(contingency_degradation=contingency_degradation, switch_cues=switch_cues)
#             file = open('/home/terra/Documents/thesis/BalancingControl/' + fname)


#         task_params = js.load(file)                                                                                 


#         np.random.seed(1)
#         colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
#         sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
#         starts = np.asarray(task_params['starts'])           # starting position of agent
#         planets = np.asarray(task_params['planets'])         # planet positions 
#         trial_type = np.asarray(task_params['trial_type'])
#         blocks = np.asarray(task_params['block'])


#         trials = starts.shape[0]                         # number of episodes
#         nblocks = int(blocks.max()+1)
#         trials = blocks.size
#         block = int(trials/nblocks)

#         all_optimal_seqs = np.unique(sequence)                                                                            

#         # define reward probabilities dependent on time and position for reward generation process 
#         Rho = np.zeros([trials, nr, ns])

#         for i, pl in enumerate(planets):
#             if i >= block*(nblocks-2) and i < block*(nblocks-1) and contingency_degradation:
#                 # print(i)
#                 # print(pl)
#                 Rho[i,:,:] = planet_reward_probs_switched[[pl]].T
#                 print(Rho[i,::])
#             else:
#                 Rho[i,:,:] = planet_reward_probs[[pl]].T


#         u = 0.99
#         utility = np.array([(1-u)/2,(1-u)/2,u])




#         reward_counts = np.ones([nr, npl, nc])

#         par_list = [h,                        \
#                     context_trans_prob,       \
#                     cue_ambiguity,            \
#                     'avg',                    \
#                     Rho,                      \
#                     utility,                  \
#                     state_transition_matrix,  \
#                     planets,                  \
#                     starts,                   \
#                     colors,
#                     reward_counts]

#         prefix = ''
#         if switch_cues == True:
#             prefix += 'switch1_'
#         else:
#             prefix +='switch0_'

#         if contingency_degradation == True:
#             prefix += 'degr1_'
#         else:
#             prefix += 'degr0_'

#         for r in range(repetitions):
#             fname = prefix +'p' + str(cue_ambiguity) + '_q' + str(context_trans_prob) + '_h' + str(h) + '_run'+str(r) + '.json'
#             run = run_agent(par_list, trials, T, ns , na, nr, nc, npl) 
        
#             fname = os.path.join(folder, fname)
#             print(fname)

#             jsonpickle_numpy.register_handlers()
#             pickled = pickle.encode(run)
#             with open(fname, 'w') as outfile:
#                 json.dump(pickled, outfile)








# filehandler = open("run_object.txt","wb")
# pickle.dump(run,filehandler)
# filehandler.close()

# ################################################
# ################################################
# #%%
# file = open("run_object.txt",'rb')
# run = pickle.load(file)
# trials = run.trials
# agent = run.agent
# perception = agent.perception
# observations = run.observations
# true_optimal = run.environment.true_optimal
# context_cues = run.environment.context_cues
# policies = run.agent.policies
# actions = run.actions[:,:3] 
# executed_policy = np.zeros(trials)

# for pi, p in enumerate(policies):
#     inds = np.where( (actions[:,0] == p[0]) & (actions[:,1] == p[1]) & (actions[:,2] == p[2]) )[0]
#     executed_policy[inds] = pi
# reward = run.rewards


# data = pd.DataFrame({'executed': executed_policy,
#                      'optimal': true_optimal,
#                      'trial': np.arange(true_optimal.size),
#                      'trial_type': trial_type})

# data['chose_optimal'] = data.executed == data.optimal
# data['optimality'] = np.cumsum(data['chose_optimal'])/(data['trial']+1)
# fig = plt.figure()
# plt.subplot(1,2,1)
# ax = sns.scatterplot(data=data[['executed','chose_optimal']])
# cols = [[0,1,1], [1,0,0],[0,1,1]] 
# ranges = data.groupby('trial_type')['trial'].agg(['min', 'max'])
# for i, row in ranges.iterrows():
#     ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# plt.subplot(1,2,2)
# ax = sns.lineplot(data=data, x=data['trial'], y=data['optimality'])


# for i, row in ranges.iterrows():
#     ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# fig.figure.savefig('1.png', dpi=300) 
# # plt.close()

# #%%
# t = -1

# posterior_context = agent.posterior_context
# data = pd.DataFrame({'posterior_h3b': posterior_context[:t,3,0],
#                      'posterior_h6o': posterior_context[:t,3,1],
#                      'posterior_h6b': posterior_context[:t,3,2],
#                      'posterior_h3o': posterior_context[:t,3,3],
#                      'context_cue': context_cues[:t],    
#                      'trial_type': trial_type[:t],
#                      'trial': np.arange(trial_type[:t].size)})


# cols = [[0,1,1], [1,0,0],[0,1,1]] 
# fig2 = sns.lineplot(data=data.iloc[:,:4])
# ranges = data.groupby('trial_type')['trial'].agg(['min', 'max'])
# for i, row in ranges.iterrows():
#     fig2.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# # fig2 = sns.scatterplot(data=data.iloc[:,4])
# # fig2.figure.savefig('2.png',dpi=300)


# #%%

# posterior_rewards = agent.posterior_dirichlet_rew
# print(posterior_rewards.shape)





# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWNNNXXXXNNXXXXXXNNWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWNNNNWWNKK00OOOO00K0O00000KKKXXNWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMWWNNWNXK000OO0K00OOkkOO0000kkO0OOO000OO000KXNWMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMWNXKK0OO00000OOO0000OxxO00000OkddkkkkOOkxdxkkOO00KNWMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMWWNNXXKK0OOkO00OOkkOKK000Okdk00OOOkxkkO000OOkdokOOOOOOOO0KNWMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMWNXK0K00000OOk0000OOkdxXKOOOkxdxOkkkkkO00OxdkOOkxdxkOOOOkOOOkO0KNMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMWNXXK00O00OkOOkkO0K0000kOKOkxdkkxkOO00OOkxolldO00OxxxkOkkdxOOkkkkkO0XWMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMWNXXXXK0Oxddddxkxllx0NK0OOOkkkocodxxxxxxdl:;:lxO0KK0OOOkkxoodkkxxxxxkxk0XWMMMMMMMMMMMMM
# MMMMMMMMMMMWNXXNWN0OkkkkkkkOOkdldOK0xddddxkdc;:cc:::;;;:cok0KXKKXXKOxdooxkkxxddxdodxkkOKNWMMMMMMMMMM
# MMMMMMMMMWNNXKOOOxodO0OOdodkkkkxdkkxdl::looll:;;:cllodxxddkKXK0KXX0kdlldkkkkxxxxxxkOkkkxkKWMMMMMMMMM
# MMMMMMMMWXKXXXK0OkdkK0OOkdoddxkkxdxdxxl;;;;:ldxkxxkkkkOOkkO0000KKOxoc;lxkkxddddolokkxxxxdx0NMMMMMMMM
# MMMMMMWX0KNWNX0OOxdkKKOkkkdllddxkxoooool::ldxxkkdxxxkkO0kkOK0Okkdoc;:oxOkxdolc:,;dOkxxxddkk0WMMMMMMM
# MMMMMNK0kOXN0kxxxddOKOxdkOdc:loooolllccoxkkxdddodxxkxxkxoldxdol:;;:oxkkxxddl:;;:ldxxddddxkkxOXWMMMMM
# MMMMN0KNK0KOkkkxkkkkxdooxxdl;;cc::::;;cxkddddolloollllc;,;;;,,',:okOkxdddoocoxkxdolooodddxxxxOXWMMMM
# MMWXOkO0Okkxxxddolloooooooolc;,:cloxkxxxdddool:;;,,,,;;:::::ccldddxxxddolc:oxkxdddl:::clooddxxOXMMMM
# MWKkxdodxxdoodddl:;,:cllllccoddxO0K0kddooool:;''';:coxkOkxxkkxkkxddddolccldxxdddddoccllccldddxdkXWMM
# MXkolclxkxxdooddxxdl;,,;;:lxkkxxkxxdlcccc:;;:cooodxxxxxxddddoooloooooooddddddoloodddkkxxddxxxxxxkXMM
# W0doccodooooclllooxxdoc;;okkxxxddool:,'',;coddxxxdddlcclllcc:;;;;;;;:ccllllc::cdddddxkddddxxdddxx0WM
# Xkxkxdoolllcclllccllllldkxooddooolc;'';looooooooollc,'',,,,'',codolc:;;,',,,,;:looodxdddodddddodx0WM
# Kxooolllllcclolllccc:coddoolclc:;,,;:clollllllll:,,,;cllooooooodxddooolccldddoc::cllooooooolc:lddONM
# Nkollc:cllcclllllc::cdxoooddl:,,;:codollcc:;;;,,,:cloooolloloc;:cccccccclooollolc::clclll:;ccloddxKW
# M0lllcc:clllllcccc::lolllllooccodolllllcc;..',:cllllcccccccc:,.',,,,;;;:;;;;::ccll::ccclllldxxdddxON
# MKoclllc:clllc;,,:cclcccccllc:clllllllc:;;:cllllcccc::;;;::;,'.',,;:clllccc:;;;,,,;;::cccloooooxdoxX
# MWOocllllllc;',,;;:cccccccll::ccccccclllllllcccc:;;;;,,'.''...,;::ccllcccccccccc:;;,,,;:cclooodo:lkN
# MMWOc:cllc;'.';:;,,,:ccccccllcccccccclllcccccc:;,'''...'....',;:ccccc:::::::::::ccccc:,,;ccllll::d0W
# MMMNOlccc;..',;;;'...:lcccclllccc:;,,:ccccccc::;'...  ..''',;:::cccc:::;;:::::::::::ccc:;,;cclc:o0WM
# MMMMWOlcc:;;;;;,,'..',:cccccccccc'..,;:cccccc:;'...    ..',;::::ccc::::::::::::::::::ccl:,;::::o0WMM
# MMMMMNOollcccc:::;;:::cllllcc::,'';cccccccc:;'''.........';;;::::::::::::::::::::::::::clc::clxKWMMM
# MMMMMMWKkxdooolclllccclloolc;,'';cllllcc:,,;,.';:,''''..,;::;;;;;;::;;;;::;:::::::::::ccl:;cokXWMMMM
# MMMMMMMMMWNNXXOxdxxxxkkkOOdllllooooooll:,'.....';c::;,,;:::::::::::::;;;;;;;;;::::::::coxxxOKNMMMMMM
# MMMMMMMMMMMMMMMMWMMMMMMMMNx:;;;;,,;;;;,..........,::c::;;,,,,,,,,,,;;;;;;;;;;::::::::cdKWMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMWOc;,''...................';:c::;'...,,''.''''',;ldolccccldkKNMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMNkc;,,,'...................';;;;'.. .',,,''''''oXWNNXXXXXWMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMWXOdl:,'''............',;lddxkOxc'....',,,,;cxXMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNX0xol:;;,;;:cldxOKXNWMMMMMWXx:'.......,kWMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNXKXXNNWMMMMMMMMMMMMMMMWKko;'.....;kWMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMW0o:;;;:;:kNMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMN0dccccclkNMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNOdoollo0WMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWXkdodONMMMMMMMMMMMMMMMMMMMMMMMM


# %%

