
import numpy as ar
array = ar.asarray
import agent as agt
import perception as prc
import action_selection as asl
import inference_two_seqs as inf
import action_selection as asl
import numpy as np
import itertools
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import pandas as pd
import os
import scipy as sc
import scipy.signal as ss
import gc
from environment import PlanetWorld
from world import FakeWorld
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy

#ar.autograd.set_detect_anomaly(True)
###################################
"""load data"""

switch = 0
degr = 1
p = 0.85
learn_rew  = 1
q = 0.8
h=4
db = 6
tb = 4
tpb = 70
n_part = 1
u = [1,1, 98]
util = '-'.join([str(ut) for ut in u])
rew = 0
dec = 1
conf ='shuffled'
folder = "temp/" + 'shuffled' + '/'

run_name = "multiple_hier_switch"+str(switch) +"_degr"+str(degr) +"_p"+str(p)+ "_learn_rew"+str(learn_rew)+\
           "_q"+str(q) + "_h"+str(h)  + "_" + str(tpb) +  "_" + str(tb) + str(db) +\
           "_dec" + str(dec) +"_rew" + str(str(rew)) + "_" + "u"+ util + "_" + conf + "_extinguish.json"
print(run_name)

fname = os.path.join(folder, run_name)
jsonpickle_numpy.register_handlers()
    
with open(fname, 'r') as infile:
    loaded = json.load(infile)

worlds = pickle.decode(loaded)
world = worlds[0]

meta = worlds[-1]

data = {}
data["actions"] = world.actions
data["rewards"] = world.rewards
data["observations"] = world.observations
data["context_obs"] = world.environment.context_cues
data["planets"] = world.environment.planet_conf
data['environment'] = world.environment
# print(data['planets'][:10,:])
###################################
"""experiment parameters"""

trials =  world.trials #number of trials
T = world.T #number of time steps in each trial
ns = 6 #number of states
no = 2 #number of observations
na = 2 #number of actions
npi = na**(T-1)
nr = 3
npl = 3

learn_pol=1
learn_habit=True

    
utility = array([ut/100 for ut in u])
"""
create matrices
"""

#generating probability of observations in each state
A = ar.eye(ns)


#state transition generative probability (matrix)
nc = 4
B = ar.zeros([ns,ns,na])

m = [1,2,3,4,5,0]
for r, row in enumerate(B[:,:,0]):
    row[m[r]] = 1

j = array([5,4,5,6,2,2])-1
for r, row in enumerate(B[:,:,1]):
    row[j[r]] = 1

B = np.transpose(B, axes=(1,0,2))
B = np.repeat(B[:,:,:,None], repeats=nc, axis=3)



planet_reward_probs = array([[0.95, 0   , 0   ],
                                [0.05, 0.95, 0.05],
                                [0,    0.05, 0.95]]).T    # npl x nr

planet_reward_probs_switched = array([[0   , 0    , 0.95],
                                        [0.05, 0.95 , 0.05],
                                        [0.95, 0.05 , 0.0]]).T 


Rho = ar.zeros([trials, nr, ns])
planets = array(world.environment.planet_conf)
for i, pl in enumerate(planets):
    if i >= tpb*tb and i < tpb*(tb + db) and degr == 1:
        Rho[i,:,:] = planet_reward_probs_switched[tuple([pl])].T
    else:
        Rho[i,:,:] = planet_reward_probs[tuple([pl])].T


if learn_rew:
    C_alphas = ar.ones([nr, npl, nc])
else:
    C_alphas = np.tile(planet_reward_probs.T[:,:,None]*20,(1,1,nc))+1
    
C_agent = ar.zeros((nr, npl, nc))
for c in range(nc):
    for pl in range(npl):
        C_agent[:,pl,c] = C_alphas[:,pl,c]/ C_alphas[:,pl,c].sum() 
#         array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(npl)]).T


r = (1-q)/(nc-1)

transition_matrix_context = np.zeros([nc,nc]) + r
transition_matrix_context = transition_matrix_context - np.eye(nc)*r + np.eye(nc)*q

"""
create policies
"""

pol = array(list(itertools.product(list(range(na)), repeat=T-1)))

npi = pol.shape[0]

# prior over policies
# prior_pi = ar.ones(npi)/npi #ar.zeros(npi) + 1e-3/(npi-1)
alphas = ar.zeros((npi,nc)) + learn_pol
prior_pi = alphas / alphas[:,0].sum()
# alphas = array([learn_pol])

"""
set state prior (where agent thinks it starts)
"""

state_prior = ar.ones((ns))
state_prior = state_prior/ state_prior.sum()

prior_context = array([0.99] + [(1-0.99)/(nc-1)]*(nc-1))


no =  2                                        # number of observations (background color)
C = ar.zeros([no, nc])
dp = 0.001
p2 = 1 - p - dp
C[0,:] = array([p,dp/2,p2,dp/2])
C[1,:] = array([dp/2, p, dp/2, p2])

"""
set up agent
"""
#bethe agent

pol_par = alphas


environment = data['environment']
environment.hidden_states[:] = 0

# perception
bayes_prc = prc.HierarchicalPerception(
    A, 
    B, 
    C_agent, 
    transition_matrix_context, 
    state_prior, 
    utility, 
    prior_pi, 
    dirichlet_pol_params = pol_par,
    dirichlet_rew_params = C_alphas,
    generative_model_context=C,
    T=T,dec_temp=dec, r_lambda=rew)

ac_sel = asl.AveragedSelector(trials = trials, T = T, number_of_actions = na)

agent = agt.BayesianPlanner(bayes_prc, ac_sel, pol,
                  trials = trials, T = T,
                  prior_states = state_prior,
                  prior_policies = prior_pi,
                  number_of_states = ns,
                  prior_context = prior_context,
                  learn_habit = learn_habit,
                  learn_rew = True,
                  #save_everything = True,
                  number_of_policies = npi,
                  number_of_rewards = nr, number_of_planets = 3)


world = FakeWorld(agent, data['observations'],data['rewards'], data['actions'],trials=trials, T=T, log_prior=0)
world.context_obs = data['environment'].context_cues
world.planets = data['environment'].planet_conf
world.simulate_experiment(curr_trials = [i for i in range(30)])


data = {'policies': world.log_policies, 'contexts': world.log_context, 'prior': world.log_prior_pols, 'post': world.log_post_pols}



with open('numpy.json', 'w') as outfile:
    json.dump(data,outfile) 
