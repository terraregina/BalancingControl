### #     # ######  ####### ######  #######  #####  
 #  ##   ## #     # #     # #     #    #    #     # 
 #  # # # # #     # #     # #     #    #    #       
 #  #  #  # ######  #     # ######     #     #####  
 #  #     # #       #     # #   #      #          # 
 #  #     # #       #     # #    #     #    #     # 
### #     # #       ####### #     #    #     #####  
                                                    

                                                
import json  as js
import numpy as np
import itertools
import os
import action_selection as asl
import perception as prc
import agent as agt

from itertools import product
from environment import PlanetWorld
from agent import BayesianPlanner
from world import World

import pickle
 

#       #######    #    ######        #     #    #    ######  ###    #    ######  #       #######  #####  
#       #     #   # #   #     #       #     #   # #   #     #  #    # #   #     # #       #       #     # 
#       #     #  #   #  #     #       #     #  #   #  #     #  #   #   #  #     # #       #       #       
#       #     # #     # #     #       #     # #     # ######   #  #     # ######  #       #####    #####  
#       #     # ####### #     #        #   #  ####### #   #    #  ####### #     # #       #             # 
#       #     # #     # #     #         # #   #     # #    #   #  #     # #     # #       #       #     # 
####### ####### #     # ######           #    #     # #     # ### #     # ######  ####### #######  #####  

np.random.seed(2345)

file = open('/home/terra/Documents/thesis/BalancingControl/config_task.json')
data = js.load(file)                                                                                 

colors = np.asarray(data['context'])          # 0/1 as indicator of color
sequence = np.asarray(data['sequence'])       # what is the optimal sequence
starts = np.asarray(data['starts'])           # starting position of agent
planets = np.asarray(data['planets'])         # planet positions 

all_optimal_seqs = np.unique(sequence)

r = [[0,1]]*3
moves = np.asarray(list(product(*r)))

h1_ind = all_optimal_seqs[0]
h1_seq = moves[all_optimal_seqs[0]]
h2_seq = moves[all_optimal_seqs[1]]

                                                                                

nplanets = 6
state_transition_matrix = np.zeros([6,6,2])

m = [1,2,3,4,5,0]
for r, row in enumerate(state_transition_matrix[:,:,0]):
    row[m[r]] = 1

j = np.array([5,4,5,6,2,2])-1
for r, row in enumerate(state_transition_matrix[:,:,1]):
    row[j[r]] = 1



# reward probabiltiy vector
p = [0.95, 0.05, 0.95] #needed?
planet_rewards = [-1,0,1]
planet_reward_probs = np.array([[0.95, 0,    0,  ],
                                [0.05, 0.95, 0.05],
                                [0,    0.05, 0.95]]).T    # npl x nr



####### #     # #     #  #####  ####### ### ####### #     #       ######  ####### #######  #####  
#       #     # ##    # #     #    #     #  #     # ##    #       #     # #       #       #     # 
#       #     # # #   # #          #     #  #     # # #   #       #     # #       #       #       
#####   #     # #  #  # #          #     #  #     # #  #  #       #     # #####   #####    #####  
#       #     # #   # # #          #     #  #     # #   # #       #     # #       #             # 
#       #     # #    ## #     #    #     #  #     # #    ##       #     # #       #       #     # 
#        #####  #     #  #####     #    ### ####### #     #       ######  ####### #        #####  
                                                                                               



def run_agent(par_list, trials, T, ns=6, na=2, nr=3, nc=2, npl=2, deval=False):

    # learn_pol          = initial concentration parameters for POLICY PRIOR
    # context_trans_prob = probability of staing in a given context, int
    # avg                = average vs maximum selection, True for avg
    # Rho                = environment reward generation probability as a function of time, dims: trials x nr x ns
    # utility            = GOAL PRIOR, preference p(o)
    # B                  = state transition matrix depedenent on action: ns x ns x actions
    # npl                = number of unique planets accounted for in the reward contingency representation  
    # C_beta           = phi or estimate of p(reward|planet,context) 
    
    learn_pol, context_trans_prob, avg, Rho, utility, B, planets, starts, colors = par_list


    """
    create matrices
    """

    #generating probability of observations in each state
    A = np.eye(ns)


    # agent's initial estimate of reward generation probability
    C_beta = np.zeros((nr, npl, nc))


    # initialized to true values
    reward_counts = planet_reward_probs.T*100

    for c in range(nc):
        C_beta[:,:,c] = reward_counts.copy()
 
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
    # not currently done according to 1/h
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
    NEED TO ALSO DEFINE A DISTRIBUTION OF p(O|C) for the color
    """


    no =  np.unique(colors).size                    # number of observations (background color)
    C = np.zeros([no, nc])
    C[:] = 1/nc

    

    """
    set up agent
    """

    # perception
    bayes_prc = prc.HierarchicalPerception(A, 
                                           state_transition_matrix, 
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

    return w




#     #    #    ### #     # 
##   ##   # #    #  ##    # 
# # # #  #   #   #  # #   # 
#  #  # #     #  #  #  #  # 
#     # #######  #  #   # # 
#     # #     #  #  #    ## 
#     # #     # ### #     # 

na = 2                                           # number of unique possible actions
nc = 4                                           # number of contexts, planning and habit
nr = 3                                           # number of rewards
ns = planets.shape[1]                            # number of unique travel locations
npl = 3
steps = 3                                        # numbe of decisions made in an episode
T = steps + 1                                    # episode length
trials = starts.shape[0]                         # number of episodes

# define reward probabilities dependent on time and position for reward generation process 
Rho = np.zeros([trials, nr, ns])


for i, pl in enumerate(planets):
    Rho[i,:,:] = planet_reward_probs[[pl]].T
    if i < 3:
        print(pl)
        print(Rho[i,:,:])

context_trans_prob = 1/nc                          # probability of staying in that context

u = 0.99
utility = np.array([(1-u)/2,(1-u)/2,u])


par_list = [100,                      \
            context_trans_prob,       \
            'avg',                    \
            Rho,                      \
            utility,                  \
            state_transition_matrix,  \
            planets,                  \
            starts,                   \
            colors]

run = run_agent(par_list, trials, T, ns , na, nr, nc, npl)


filehandler = open("run_object.txt","wb")
pickle.dump(run,filehandler)
filehandler.close()

file = open("run_object.txt",'rb')
run = pickle.load(file)
file.close()


















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