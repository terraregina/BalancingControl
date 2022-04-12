
import torch as tr
array = tr.tensor
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
# import distributions as analytical_dists

from tqdm import tqdm
import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl

class SingleInference(object):

    def __innit__(self, agent, data):
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data



    def model(self):

        alpha_h = pyro.param('alpha_h', lambda: tr.ones(1))
        beta_h = pyro.param('beta_h', lambda: tr.ones(1))

        # sample initial vaue of parameter from Beta distribution
        h = pyro.sample('h', dist.Beta(alpha_h, beta_h))


        self.agent.reset(param_dict)
        #self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
        
        for tau in range(self.trials):
            for t in range(self.T):
                
                if t==0:
                    prev_response = None
                    context = None
                else:
                    prev_response = self.data["actions"][tau, t-1]
                    context = None
        
                observation = self.data["observations"][tau, t]
        
                reward = self.data["rewards"][tau, t]
        
                self.agent.update_beliefs(tau, t, observation, reward, prev_response, context)
        
                if t < self.T-1:
                
                    probs = self.agent.perception.posterior_actions[-1]
                    #print(probs)
                    if ar.any(ar.isnan(probs)):
                        print(probs)
                        print(dec_temp, lamb_pi, lamb_r)
            
                    curr_response = self.data["actions"][tau, t]
                    #print(curr_response)
                    # print(tau, t, probs, curr_response)
                    #print(tau,t,param_dict)
                    
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.T), obs=curr_response)
                    


    def guide(self):
        pass



    def infer_posterior(self, iter_steps=1000, num_particles=10, optim_kwarfs={'1r': 0.01}):
        pass