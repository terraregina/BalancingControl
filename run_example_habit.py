import os
import sys
from itertools import product, repeat
import json
import pickle
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy

import numpy as np
import torch as ar


import perception as prc
import agent as agt
from environment import PlanetWorld
from world import World, FittingWorld, GroupWorld
import action_selection as asl


import seaborn as sns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Pool
import tqdm
import gc

from sim_parameters import *

gc.enable()




def run_agent(par_list, trials, T, ns=6, na=2, nr=3, nc=2, npl=2, trial_type=None, true_optimal=None, use_fitting=False):

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

    # p = context_trans_prob
    # q = (1-p)/(nc-1)

    # transition_matrix_context = np.zeros([nc,nc]) + q
    # transition_matrix_context = transition_matrix_context - np.eye(nc)*q + np.eye(nc)*p 
    # print('\n\nnormal matrix\n', transition_matrix_context)

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
    create policies and setup concentration parameters
    The pseudo counts alpha^t_ln which parameterize the prior over actions for
    """

    pols = np.array(list(product(list(range(na)), repeat=T-1)))
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
    
    ################
    # p = cue_ambiguity                               # how strongly agent associates context observation with a particular context       
    # dp = 0.001
    # p2 = 1 - p - dp
    # C[0,:] = [p,dp/2,p2,dp/2]
    # C[1,:] = [dp/2, p, dp/2, p2]
    # print('\noriginal model observations')
    # print(C)


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
        dec_temp_cont = ar.tensor([dec_temp_cont])
        planets = ar.tensor(planets, dtype=ar.long)

        ac_sel = asl.FittingAveragedSelector(trials = trials, T = T,
                                      number_of_actions = na)

        bayes_prc = prc.GroupFittingPerception(\
                               A, 
                               B, 
                               C_agent, 
                               state_prior, 
                               utility, 
                               prior_pi,
                               pols,
                               alpha_0,
                               C_beta,
                               transition_matrix_context = transition_matrix_context, 
                               generative_model_context = C, prior_context=prior_context,
                               number_of_planets = npl,
                               T=T,dec_temp=dec_temp, dec_temp_cont=dec_temp_cont, trials=trials)
    
        bayes_pln = agt.FittingAgent(bayes_prc,
                                    ac_sel,
                                    pols,
                                    prior_states = state_prior,
                                    prior_policies = prior_pi,
                                    prior_context = prior_context, number_of_planets = npl,
                                    number_of_policies=npi, number_of_rewards=nr,trials=trials, T=T,\
                                    number_of_states = ns)

    else:

        raise Exception('Trying to run numpy agent from wrong file!\nTo simulate with the numpy agent or alternatively the torch agent without an extra dimension for the participant use the "run_example_habit_old.py" file')
    
    #     ac_sel = asl.AveragedSelector(trials = trials, T = T,
    #                                   number_of_actions = na)

    #     bayes_prc = prc.HierarchicalPerception(A, 
    #                            B, 
    #                            C_agent, 
    #                            transition_matrix_context, 
    #                            state_prior, 
    #                            utility, 
    #                            prior_pi,
    #                            pols,
    #                            C_alphas,
    #                            C_beta,
    #                            trials= trials,
    #                            generative_model_context = C,
    #                            init_planets = planets[0],
    #                            T=T,dec_temp=dec_temp, dec_temp_cont=dec_temp_cont,\
    #                            possible_rewards = possible_rewards, r_lambda=rew)

    #     bayes_pln = agt.BayesianPlanner(bayes_prc,
    #                                     ac_sel,
    #                                     pols,
    #                                     prior_states = state_prior,
    #                                     prior_policies = prior_pi,
    #                                     trials = trials,
    #                                     prior_context = prior_context,            
    #                                     learn_habit = True,
    #                                     learn_rew = True,
    #                                     number_of_planets = npl,
    #                                     number_of_rewards = nr)
    
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
                                    
    w = GroupWorld(environment, bayes_pln, trials = trials, T = T)

    """
    create world
    """

    bayes_pln.world = w

    if not trial_type is None:
        bayes_pln.trial_type = trial_type
        w.trial_type = trial_type
        bayes_pln.true_optimal = true_optimal
    else:
        raise('Trial type and true optimal not passed')

    """
    simulate experiment
    """

    w.simulate_experiment(range(trials))
    # w.simulate_experiment(range(2))

    w.h = learn_pol
    w.q = context_trans_prob
    w.p = cue_ambiguity
    w.dec_temp = dec_temp
    
    if use_fitting:
        wn = w
        wn.actions = np.array(wn.actions)
        wn.rewards = np.array(wn.rewards)
        wn.observations = np.array(wn.observations)
        for key in wn.environment.__dict__.keys():
            if ar.is_tensor(w.environment.__dict__[key]):
                w.environment.__dict__[key] = np.array(w.environment.__dict__[key])

        keys = ['actions', 'observations', 'rewards', 'posterior_actions',\
                'posterior_rewards', 'context_obs', 'policies','possible_polcies','prior_states','prior_context',\
                # 'prior_policies','posterior_contexts','control_probs','planets','prev_pols','possible_policies']
                'prior_policies','posterior_contexts' ,'control_probs']

        for key in keys:
            wn.agent.__dict__[key] =  np.array(wn.agent.__dict__[key])
        
        keys = ['rewards', 'observations', 'dirichlet_rew_params', 'dirichlet_pol_params', 'bwd_messages', 'fwd_messages', 'obs_messages', 'rew_messages', 'fwd_norms',\
                'curr_gen_mod_rewards', 'posterior_states', 'posterior_policies', 'posterior_actions',\
                'posterior_contexts', 'likelihoods'] 

        for key in keys:
            wn.agent.perception.__dict__[key] = np.array(ar.stack(wn.agent.perception.__dict__[key]))


        keys = ['big_trans_matrix', 'generative_model_observations','generative_model_states',\
                'generative_model_context','transition_matrix_context','prior_rewards','prior_states',\
                'dec_temp','dec_temp_cont','policies','actions','alpha_0','dirichlet_rew_params_init',\
                'dirichlet_pol_params_init','prior_context'] # ,'context_obs_surprise', 'outcome_suprise','policy_entropy', 'policy_surprise']
        
        for key in keys:
            wn.agent.perception.__dict__[key] =  np.array(wn.agent.perception.__dict__[key])

        # wn.dec_temp = 2
        wn.agent.perception.generative_model_rewards =  0
        wn.agent.perception.prior_policies =  0
        # wn.agent.perception.possible_rewards =  0
        # wn.agent.perception.planets =  0
        wn.agent.action_selection.control_probability =  0
        # wn.agent.perception.big_trans_matrix = 0
        return wn
    else:
        return w


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
                    repetitions, use_fitting):
    
    #  unpack simulation parameters
    switch_cues, contingency_degradation, reward_naive, context_trans_prob, cue_ambiguity, h,\
    training_blocks, degradation_blocks, trials_per_block, dec_temp, dec_temp_cont,\
    rew, rewards, util, config_folder = lst
    
    # load file with task setup 
    config = 'planning_config' + '_degradation_'+ str(int(contingency_degradation)) \
                      + '_switch_' + str(int(switch_cues))                \
                      + '_train' + str(training_blocks)                   \
                      + '_degr' + str(degradation_blocks)                 \
                      + '_n' + str(trials_per_block)+ '_nr_' + str(nr) + '.json'

    folder = os.path.join(os.getcwd(),'config/' + config_folder)

    if sys.platform == "win32":
        folder = folder.replace('/','\\')
    file = open(os.path.join(folder,config))

    task_params = json.load(file)                                                                                 
    colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
    sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
    starts = np.asarray(task_params['starts'])           # starting position of agent
    planets = np.asarray(task_params['planets'])         # planet positions 
    trial_type = np.asarray(task_params['trial_type'])
    blocks = np.asarray(task_params['block'])

    nblocks = int(blocks.max()+1)                        # number of blocks
    trials = blocks.size                                 # number of trials
    block = task_params['trials_per_block']              # trials per block
 

   # setup simulation meta data
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
        'trials_per_block': task_params['trials_per_block'],
        'exp_reward': task_params['exp_reward'],
        'nr': task_params['nr'],
        'state_transition_matrix': task_params['state_transition_matrix'][0],
        'deterministic_reward': deterministic_reward
    }

    # reward probability schedule dependent on trial and planet constelation

    Rho = np.zeros([trials, nr, ns])
    if deterministic_reward:
        planet_reward_probs = planet_reward_probs.round(1)
        planet_reward_probs_switched = planet_reward_probs_switched.round(1)

    for i, pl in enumerate(planets):
        if i >= block*meta['training_blocks'] and i < block*(meta['training_blocks'] + meta['degradation_blocks']) and contingency_degradation:
            Rho[i,:,:] = planet_reward_probs_switched[tuple([pl])].T
        else:
            Rho[i,:,:] = planet_reward_probs[tuple([pl])].T

    utility = np.array([float(u)/100 for u in util])

    if reward_naive==True:
        reward_counts = np.ones([nr, npl, nc])
    else:
        reward_counts = np.ones([nr, npl, nc])
        reward_counts[:,:,:2] = np.tile(planet_reward_probs.T[:,:,np.newaxis]*init_reward_count_bias,(1,1,2))+1
        print('\nDoing different naive rewards')
        print(reward_counts)

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
                1,
                dec_temp,
                dec_temp_cont,
                rew,
                rewards]

    prefix = 'multiple_'

    if use_fitting == True:
        prefix += 'fitt_'
    else:
        prefix +='hier_'

    if switch_cues == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if contingency_degradation == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'

    fname = prefix +'p' + str(cue_ambiguity) +'_learn_rew' + str(int(reward_naive == True)) + '_q' + str(context_trans_prob) + '_h' + str(h)+ '_' +\
    str(meta['trials_per_block']) +'_'+str(meta['training_blocks']) + str(meta['degradation_blocks']) +\
    '_decp' + str(dec_temp)+ '_decc' + str(dec_temp_cont)+ '_rew' + str(rew) + \
    '_u' +  '-'.join(util) + '_' + str(nr) + '_' + config_folder + '.json'

    # check if task setup state transition matrix matches the desired simulation state transition matrix
    trial_file_stm = np.array(meta['state_transition_matrix'])
    if not np.all(trial_file_stm[:,:,1] == state_transition_matrix[:,:,1,0].T):
        raise Exception('desired state transition matrix does not match state transition matrix in trial file\n')
    
    worlds = [run_agent(par_list, trials, T, ns , na, nr, nc, npl, trial_type, sequence, use_fitting) for _ in range(repetitions)]

    meta['trial_type'] = task_params['trial_type']
    meta['optimal_sequence'] = task_params['sequence']
    worlds.append(meta)
    fname = os.path.join(os.path.join(os.getcwd(),'temp' + '/' + config_folder), fname)

    if sys.platform == "win32":
        fname = fname.replace('/','\\')
    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(worlds)
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)

    return fname


def pooled(arrays,seed=521312,repetitions=1, data_folder='temp',check_missing = False,debugging=False, use_fitting=False):

    np.random.seed(seed)
    ar.manual_seed(seed)

    lst = []
    existing_files = []

    for conf_folder in arrays[-1]:
        path = os.path.join(os.getcwd(), data_folder + '/' + conf_folder)
        existing_files += os.listdir(path)

    for i in product(*arrays):
        lst.append(list(i))

    names = []

    for li, l in enumerate(lst):
        prefix = 'multiple_'

        if use_fitting == True:
            prefix += 'fitt_'
        else:
            prefix += 'hier_'
        if l[0] == True:
            prefix += 'switch1_'
        else:
            prefix +='switch0_'

        if l[1] == True:
            prefix += 'degr1_'
        else:
            prefix += 'degr0_'


        l[13] = [str(entry) for entry in l[13]]
        fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
        str(l[8]) + '_' + str(l[6]) + str(l[7]) + \
        '_decp' + str(l[9]) + '_decc' + str(l[10]) +'_rew' + str(l[11]) + '_' + 'u'+  '-'.join(l[13])+'_' + str(nr) + '_' + l[14]
        print(l[9])

        names.append([li, fname])

    if check_missing:
        missing_files = []
        for name in names:
            if not name[1] in existing_files:
                # print(name)
                missing_files.append(name[0])

        lst = [lst[i] for i in missing_files]

    print('simulations to run: ' + str(len(lst)))

    ca = [ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs,\
        planet_reward_probs_switched,repetitions,use_fitting]

    if debugging:
        for l in [lst[0]]:
            run_single_sim(l, ca[0], ca[1], ca[2], ca[3], ca[4], ca[5], ca[6], ca[7], ca[8], ca[9], ca[10])
    else: 
        with Pool() as pool:
            for _ in tqdm.tqdm(pool.istarmap(run_single_sim, zip(lst,\
                                                    repeat(ca[0]),\
                                                    repeat(ca[1]),\
                                                    repeat(ca[2]),\
                                                    repeat(ca[3]),\
                                                    repeat(ca[4]),\
                                                    repeat(ca[5]),\
                                                    repeat(ca[6]),\
                                                    repeat(ca[7]),\
                                                    repeat(ca[8]),\
                                                    repeat(ca[9]),\
                                                    repeat(ca[10]))),
                                                    total=len(lst)):
                pass
    
    

if __name__ == '__main__':
    data_folder = 'temp'
    
    for con in conf:
        path = os.path.join(data_folder, con)
        if not os.path.exists(path):
            os.makedirs(path)

    arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
            training_blocks, degradation_blocks, trials_per_block,dec_temps, dec_context, rews, rewards, utility, conf]
    pooled(arrays,seed=34242, repetitions=repetitions,check_missing=False,debugging=debugging, use_fitting=use_fitting)





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

