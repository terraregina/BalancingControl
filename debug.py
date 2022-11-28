
#%%
import torch as ar
array = ar.tensor

import pyro
import pyro.distributions as dist
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
#%%
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
data["actions"] = ar.tensor(world.actions)
data["rewards"] = ar.tensor(world.rewards)
data["observations"] = ar.tensor(world.observations)
data["context_obs"] = ar.tensor(world.environment.context_cues)
data["planets"] = ar.tensor(world.environment.planet_conf)
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

utility = []

# ut = [0.999]
# for u in ut:
#     utility.append(ar.zeros(nr))
#     for i in range(0,nr):
#         utility[-1][i] = (1-u)/(nr-1)#u/nr*i
#     utility[-1][-1] = u
    
# utility = utility[-1]

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

Rho = ar.as_tensor(Rho)

if learn_rew:
    C_alphas = ar.ones([nr, npl, nc])
else:
    C_alphas = ar.as_tensor(np.tile(planet_reward_probs.T[:,:,None]*20,(1,1,nc))+1)
    
C_agent = ar.zeros((nr, npl, nc))
for c in range(nc):
    for pl in range(npl):
        C_agent[:,pl,c] = C_alphas[:,pl,c]/ C_alphas[:,pl,c].sum() 
#         array([(C_alphas[:,i,c])/(C_alphas[:,i,c]).sum() for i in range(npl)]).T


r = (1-q)/(nc-1)
    
transition_matrix_context = np.zeros([nc,nc],dtype='float64') + r
transition_matrix_context = ar.as_tensor(transition_matrix_context - np.eye(nc, dtype='float64')*r + np.eye(nc,dtype='float64')*q)

"""
create policies
"""

pol = array(list(itertools.product(list(range(na)), repeat=T-1)))

npi = pol.shape[0]

# prior over policies
# prior_pi = ar.ones(npi)/npi #ar.zeros(npi) + 1e-3/(npi-1)
alphas = ar.zeros((npi,nc)) + learn_pol
prior_pi = alphas / alphas[:,0].sum()
alphas = array([learn_pol])

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
bayes_prc = prc.FittingPerception(
    A, 
    B, 
    C_agent, 
    transition_matrix_context, 
    state_prior, 
    utility, 
    prior_pi, 
    pol,
    pol_par,
    C_alphas,
    generative_model_context=C,
    T=T, 
    trials=trials,npart=n_part,
    dec_temp = array([dec]), r_lambda=rew)

ac_sel = asl.AveragedSelector(trials = trials, T = T, number_of_actions = na)

agent = agt.FittingAgent(bayes_prc, ac_sel, pol,
                  trials = trials, T = T,
                  prior_states = state_prior,
                  prior_policies = prior_pi,
                  number_of_states = ns,
                  prior_context = prior_context,
                  learn_habit = learn_habit,
                  learn_rew = True,
                  #save_everything = True,
                  number_of_policies = npi,
                  number_of_rewards = nr)


world = FakeWorld(agent, data['observations'],data['rewards'], data['actions'],trials=trials, T=T, log_prior=0)
world.context_obs = ar.tensor(data['environment'].context_cues)
world.planets = ar.tensor(data['environment'].planet_conf)
world.simulate_experiment()#curr_trials = [i for i in range(30)])

data = {'policies': world.log_policies, 'contexts': world.log_context, 'prior': world.log_prior_pols, 'post': world.log_post_pols}


# with open('torch.json', 'w') as outfile:
#     json.dump(data,outfile) 
# #%%

# import pandas as pd
# import json

# with open('torch.json', 'r') as infile:
#     torch_data  = json.load(infile) 

# with open('numpy.json', 'r') as infile:
#     numpy_data  = json.load(infile) 

# df1 = pd.DataFrame(numpy_data)
# df2 = pd.DataFrame(torch_data)


# df1['same'] = df1['contexts'] == df2['contexts']
# df1['same_pols'] = df1['policies'] == df2['policies']
# df1['same_cont'] = df1['contexts'] == df2['contexts']
# df1['same_pols'].sum()
# df1['same_cont'].sum()
# df1['context_df2'] = df2['contexts']
# df1['policies_df2'] = df2['policies']
# df1['prior_df2'] = df2['prior']
# df1['post_df2'] = df2['post']
# df1[df1['same'] ==  False][['prior','prior_df2', 'post', 'post_df2']]




# %%