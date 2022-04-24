
from logging import raiseExceptions
from re import S
import torch as ar
array = ar.tensor
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

device = ar.device("cpu")


class SingleInference(object):

    def __init__(self, agent, data):
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data



    def model(self):

        alpha_h = ar.ones(1).to(device)
        beta_h = ar.ones(1).to(device)

        # sample initial vaue of parameter from Beta distribution
        h = pyro.sample('h', dist.Beta(alpha_h, beta_h))
        param_dict = {"h": h}

        self.agent.reset(param_dict)
        #self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
        
        for tau in range(self.trials):
            for t in range(self.T):
                
                if t==0:
                    prev_response = None
                else:
                    prev_response = self.data["actions"][tau, t-1]
                context = self.data['context_obs'][tau]
                    # context = None
        
                observation = self.data["observations"][tau, t]
                reward = self.data["rewards"][tau, t] 
                self.agent.planets = self.data["planets"][tau]
                self.agent.update_beliefs(tau, t, observation, reward, prev_response, context)
        
                if t < self.T-1:
                    probs = self.agent.perception.posterior_actions[-1]
                    #print(probs)
                    if ar.any(ar.isnan(probs)):
                        raiseExceptions('HAD NANS!')
                        print('\nhad nan in actions probs:')
                        print(probs)
                        print(h)
            
                    curr_response = self.data["actions"][tau, t]
                    # print('planets: ', self.agent.planets)
                    # print('observation: ', observation)
                    # print('inferred state:', self.agent.perception.posterior_states[-1][:,t,:,0,0])
                    # print(probs)
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.T), obs=curr_response)
                    


    def guide(self):

        alpha_h = pyro.param("alpha_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        beta_h = pyro.param("beta_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # sample initial vaue of parameter from Beta distribution
        h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)
  
        param_dict = {"alpha_h": alpha_h, "beta_h": beta_h, "h": h}

        return param_dict

    def infer_posterior(self,
                        iter_steps=1000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}):

        """
        Perform SVI over free model parameters.
        """

        pyro.clear_param_store()

        svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))

        loss = []
        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            loss.append(ar.tensor(svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            alpha_h = pyro.param("alpha_h").data.numpy()
            beta_h = pyro.param("beta_h").data.numpy()
            print("alpha: ", alpha_h, " beta: ", beta_h)
            print('h: ', (alpha_h+beta_h)/alpha_h)        
            # if ar.isnan(loss[-1]):
                # break

        self.loss = [l.cpu() for l in loss]
        
        param_dict = {"alpha_h": alpha_h, "beta_h": beta_h, "h":h}
        
        return self.loss, param_dict