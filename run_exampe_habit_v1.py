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
from itertools import product, repeat
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import time
from multiprocessing import Pool

import perception as prc
import agent as agt
from environment import PlanetWorld, FittingPlanetWorld
from agent import BayesianPlanner
from world import World, FittingWorld

import torch as ar
#%%

####### #     # #     #  #####  ####### ### ####### #     #       ######  ####### #######  #####  
#       #     # ##    # #     #    #     #  #     # ##    #       #     # #       #       #     # 
#       #     # # #   # #          #     #  #     # # #   #       #     # #       #       #       
#####   #     # #  #  # #          #     #  #     # #  #  #       #     # #####   #####    #####  
#       #     # #   # # #          #     #  #     # #   # #       #     # #       #             # 
#       #     # #    ## #     #    #     #  #     # #    ##       #     # #       #       #     # 
#        #####  #     #  #####     #    ### ####### #     #       ######  ####### #        #####  
                                                                                               



def run_agent(par_list, trials, T, ns=6, na=2, nr=3, nc=2, npl=2, added=None, use_fitting=False):

    # learn_pol          = initial concentration parameters for POLICY PRIOR
    # context_trans_prob = probability of staing in a given context, int
    # avg                = average vs maximum selection, True for avg
    # Rho                = environment reward generation probability as a function of time, dims: trials x nr x ns
    # utility            = GOAL PRIOR, preference p(o)
    # B                  = state transition matrix depedenent on action: ns x ns x actions
    # npl                = number of unique planets accounted for in the reward contingency representation  
    # C_beta           = phi or estimate of p(reward|planet,context) 
    
    learn_pol, context_trans_prob, cue_ambiguity, avg,\
    Rho, utility, B, planets, starts, colors, rc, learn_rew,\
    dec_temp, dec_temp_cont, rew, possible_rewards = par_list


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

    transition_matrix_context = np.zeros([nc
    
    
    ,nc]) + q
    transition_matrix_context = transition_matrix_context - np.eye(nc)*q + np.eye(nc)*p 
    
    # ///////// 
    p = context_trans_prob
    q = (1 - p)/6

    transition_matrix_context = np.array([\
        [p,    q*4,  q,    q  ],
        [q*4,  p,    q,    q  ],
        [q,    q,    p,    q*4],
        [q,    q,    q*4,  p  ]
    ])
    print('\n\nbiased transition matrix\n', transition_matrix_context)
    """ 
    create environment class
    """



    """ 
    create policies and setup concentration parameters
    The pseudo counts alpha^t_ln which parameterize the prior over actions for
    """

    pols = np.array(list(itertools.product(list(range(na)), repeat=T-1)))
    npi = pols.shape[0]


    C_alphas = np.zeros((npi, nc)) + learn_pol
    prior_pi = C_alphas / C_alphas.sum(axis=0)
    alpha_0 = learn_pol
    """
    set state prior
    """

    state_prior = np.ones((ns))
    state_prior = state_prior/ state_prior.sum() 


    """
    set action selection method
    """

    # if avg:
    #     ac_sel = asl.FittingAveragedSelector(trials = trials, T = T,
    #                                   number_of_actions = na)

    #     ac_sel = asl.AveragedSelector(trials = trials, T = T,
    #                                   number_of_actions = na)

    
    # else:

    #     ac_sel = asl.MaxSelector(trials = trials, T = T,
    #                                   number_of_actions = na)

    """
    set context prior
    """

    # prior_context = np.zeros((nc)) + 0.1/(nc-1)
    # prior_context[0] = 0.9


    prior_context = np.zeros((nc)) + 0.1/(nc-2)
    prior_context[:2] = (1 - 0.1)/2

    print('\nprior_context', prior_context)
    
    
    """
    define generative model for context
    """


    no =  np.unique(colors).size                    # number of observations (background color)
    C = np.zeros([no, nc])

    dp = 0.001
    p = cue_ambiguity                               # how strongly agent associates context observation with a particular context       
    p2 = 1 - p
    p -= dp/2
    p2 -= dp/2
   
    C[0,:] = [p,dp/2,p2,dp/2]
    C[1,:] = [dp/2, p, dp/2, p2]
    print('\ngenerative model observations')
    print(C)

    # p = cue_ambiguity                               # how strongly agent associates context observation with a particular context       
    # dp = 0.001
    # p2 = 1 - p - dp
    # C[0,:] = [p,dp/2,p2,dp/2]
    # C[1,:] = [dp/2, p, dp/2, p2]
    """
    set up environment
    """




    """
    set up agent
    """

    # perception

    nr = len(possible_rewards)

    if use_fitting == True:
        A = ar.tensor(A)
        B = ar.tensor(B)
        Rho = ar.tensor(Rho)
        planets  = ar.tensor(planets)
        starts = ar.tensor(starts)
        colors = ar.tensor(colors)
        C_agent = ar.tensor(C_agent)
        transition_matrix_context =  ar.tensor(transition_matrix_context) 
        state_prior =  ar.tensor(state_prior) 
        utility =  ar.tensor(utility) 
        prior_pi =  ar.tensor(prior_pi)
        C_alphas =  ar.tensor(C_alphas)
        C_beta =  ar.tensor(C_beta)
        C = ar.tensor(C)    
        pols = ar.tensor(pols)
        prior_context = ar.tensor(prior_context)
        alpha_0 = ar.tensor([alpha_0])
        dec_temp = ar.tensor([dec_temp])
        dec_temp_cont = ar.tensor([dec_temp])


    if use_fitting ==True:

        ac_sel = asl.FittingAveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)

        environment = FittingPlanetWorld(A,
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

        bayes_prc = prc.FittingPerception(A, 
                               B, 
                               C_agent, 
                               transition_matrix_context, 
                               state_prior, 
                               utility, 
                               prior_pi,
                               pols,
                               alpha_0,
                               C_beta,
                               generative_model_context = C,
                               T=T,dec_temp=dec_temp, dec_temp_cont=dec_temp_cont, trials=trials)
    
        bayes_pln = agt.FittingAgent(bayes_prc,
                                    ac_sel,
                                    pols,
                                    prior_states = state_prior,
                                    prior_policies = prior_pi,
                                    trials = trials,
                                    prior_context = prior_context,
                                    learn_habit=True,
                                    learn_rew = learn_rew,T=T
                                    )
        w = FittingWorld(environment, bayes_pln, trials = trials, T = T)

    else:
        ac_sel = asl.AveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)

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
                               T=T,dec_temp=dec_temp, dec_temp_cont=dec_temp_cont,\
                               possible_rewards = possible_rewards, r_lambda=rew)

        bayes_pln = agt.BayesianPlanner(bayes_prc,
                                        ac_sel,
                                        pols,
                                        prior_states = state_prior,
                                        prior_policies = prior_pi,
                                        trials = trials,
                                        prior_context = prior_context,
                                        learn_habit=True,
                                        learn_rew = learn_rew,
                                        number_of_planets = npl,
                                        number_of_rewards = nr
                                        )
                                        
        w = World(environment, bayes_pln, trials = trials, T = T)


    """
    create world
    """

    bayes_pln.world = w

    if not added is None:
        bayes_pln.trial_type = added[0]
        w.trial_type = added[0]
        bayes_pln.true_optimal = added[1]
    else:
        raise('Agent not extinguishing during reward')

    """
    simulate experiment
    """
    w.simulate_experiment(range(trials))
    w.h = learn_pol
    w.q = context_trans_prob
    w.p = cue_ambiguity
    w.dec_temp = dec_temp

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
                    repetitions):

    switch_cues, contingency_degradation, reward_naive, context_trans_prob, cue_ambiguity, h,\
    training_blocks, degradation_blocks, trials_per_block = lst
    
    config = 'config/config' + '_degradation_'+ str(int(contingency_degradation)) \
                      + '_switch_' + str(int(switch_cues))                \
                      + '_train' + str(training_blocks)                   \
                      + '_degr' + str(degradation_blocks)                 \
                      + '_n' + str(trials_per_block)+'.json'

    folder = os.getcwd()

    file = open(os.path.join(folder,config))

    task_params = js.load(file)                                                                                 

    colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
    sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
    starts = np.asarray(task_params['starts'])           # starting position of agent
    planets = np.asarray(task_params['planets'])         # planet positions 
    trial_type = np.asarray(task_params['trial_type'])
    blocks = np.asarray(task_params['block'])


    nblocks = int(blocks.max()+1)         # number of blocks
    trials = blocks.size                  # number of trials
    block = task_params['trials_per_block']           # trials per block
 
    meta = {
        'trial_file' : config, 
        'trial_type' : trial_type,
        'switch_cues': switch_cues == True,
        'contingency_degradation' : contingency_degradation == True,
        'learn_rew' : reward_naive == True,
        'context_trans_prob': context_trans_prob,
        'cue_ambiguity' : cue_ambiguity,
        'h' : h,
        'optimal_sequence' : sequence,
        'blocks' : blocks,
        'trials' : trials,
        'nblocks' : nblocks,
        'degradation_blocks': task_params['degradation_blocks'],
        'training_blocks': task_params['training_blocks'],
        'interlace': task_params['interlace'],
        'contingency_degrdataion': task_params['contingency_degradation'],
        'switch_cues': task_params['switch_cues'],
        'trials_per_block': task_params['trials_per_block']
    }

    all_optimal_seqs = np.unique(sequence)                                                                            

    # reward probabilities schedule dependent on trial and planet constelation
    Rho = np.zeros([trials, nr, ns])

    for i, pl in enumerate(planets):
        if i >= block*meta['training_blocks'] and i < block*(meta['training_blocks'] + meta['degradation_blocks']) and contingency_degradation:
            # print(i)
            # print(pl)
            Rho[i,:,:] = planet_reward_probs_switched[tuple([pl])].T
        else:
            Rho[i,:,:] = planet_reward_probs[tuple([pl])].T

    u = 0.99
    utility = np.array([(1-u)/2,(1-u)/2,u])

    if reward_naive==True:
        reward_counts = np.ones([nr, npl, nc])
    else:
        reward_counts = np.tile(planet_reward_probs.T[:,:,np.newaxis]*20,(1,1,nc))+1
        # reward_counts = np.ones([nr, npl, nc])
        # reward_counts[:,:,:2] = np.tile(planet_reward_probs.T[:,:,np.newaxis]*10,(1,1,2))+1
        # print('\nDoing different naive rewards')
        # print(reward_counts)

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
                1]

    prefix = ''
    if switch_cues == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if contingency_degradation == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'
    fname = prefix +'p' + str(cue_ambiguity) +'_learn_rew' + str(int(reward_naive == True)) + '_q' + str(context_trans_prob) + '_h' + str(h)+ '_' +\
    str(meta['trials_per_block']) +'_'+str(meta['training_blocks']) + str(meta['degradation_blocks']) + '.json'

    worlds = [run_agent(par_list, trials, T, ns , na, nr, nc, npl,trial_type=trial_type) for _ in range(repetitions)]

    worlds.append(meta)
    if False:
        fname = os.path.join(os.path.join(folder,'data'), fname)
        jsonpickle_numpy.register_handlers()
        pickled = pickle.encode(worlds)
        with open(fname, 'w') as outfile:
            json.dump(pickled, outfile)

        return fname

""""""
def main(create_configs = False):

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
# config_degradation_1_switch_1_train4_degr4_n60.json'
    h =  [1,70,100]
    cue_ambiguity = [0.6]                      # cue ambiguity refers to how certain agent is a given observation refers to a given context
    context_trans_prob = [1/nc]                # the higher the value the more peaked the distribution is
    degradation = [True]                       # bit counter intuitive, should change the name :D
    cue_switch = [False]
    reward_naive = [False]
    training_blocks = [4]
    degradation_blocks=[4]
    trials_per_block=[70]
    arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
              training_blocks, degradation_blocks, trials_per_block]

    data_path = os.path.join(os.getcwd(),'data')
    if not os.path.isdir(data_path): 
        os.mkdir(data_path)

    # needs to be run only the first time you use main
    # if create_configs:
    #     from planet_sequences import generate_trials_df
    #     create_config_files(training_blocks, degradation_blocks, trials_per_block)

    lst = []
    for i in product(*arrays):
        lst.append(list(i))

    fname = 'config/config_degradation_0_switch_0_train4_degr4_n60.json'
    # failed efforts at parallel running of sims
    ca = [ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs,\
          planet_reward_probs_switched,repetitions]

    # start =  time.perf_counter()
    # with Pool() as pool:
    #     M = pool.starmap(run_single_sim, zip(lst,\
    #                                     repeat(ca[0]),\
    #                                     repeat(ca[1]),\
    #                                     repeat(ca[2]),\
    #                                     repeat(ca[3]),\
    #                                     repeat(ca[4]),\
    #                                     repeat(ca[5]),\
    #                                     repeat(ca[6]),\
    #                                     repeat(ca[7]),\
    #                                     repeat(ca[8]),\
    #                                     repeat(ca[9])))
    # finish = time.perf_counter()
    # print(finish-start)

    start =  time.perf_counter()
    for l in lst[:1]:
        run_single_sim(l, ca[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7], ca[8], ca[9])
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s) for sequential')




# ################################################

# if __name__ == '__main__':
#     main()









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

