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
import torch as ar

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
    
    ################
    # p = cue_ambiguity                               # how strongly agent associates context observation with a particular context       
    # dp = 0.001
    # p2 = 1 - p - dp
    # C[0,:] = [p,dp/2,p2,dp/2]
    # C[1,:] = [dp/2, p, dp/2, p2]
    # print('\noriginal model observations')
    # print(C)
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
        dec_temp_cont = ar.tensor([dec_temp_cont])

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
    # w.simulate_experiment(range(4)) 

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
                'prior_policies','posterior_contexts','control_probs','planets','prev_pols','possible_policies']

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
                'dirichlet_pol_params_init','prior_context']
        
        for key in keys:
            wn.agent.perception.__dict__[key] =  np.array(wn.agent.perception.__dict__[key])

        #  FIX properlt
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

