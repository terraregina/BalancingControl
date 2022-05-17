
from logging import raiseExceptions
# from re import S
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
import distributions as analytical_dists

device = ar.device("cpu")

ar.set_num_threads(1)
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
        
        
        # concentration_dec_temp = ar.tensor(1.).to(device)
        # rate_dec_temp = ar.tensor(0.5).to(device)
        # # sample initial vaue of parameter from normal distribution
        # dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)

        # # param_dict = {"h": h, "dec_temp":dec_temp}
        # param_dict = {"dec_temp":dec_temp}
        param_dict = {"h":h}
        self.agent.reset(param_dict)

        
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
                        # print(h)
            
                    curr_response = self.data["actions"][tau, t]
                    # if(tau==self.trials-1):
                    #     print(probs)
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.T), obs=curr_response)
                    


    def guide(self):

        alpha_h = pyro.param("alpha_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        beta_h = pyro.param("beta_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
        # sample initial vaue of parameter from Beta distribution
        h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)

        # concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#interval(0., 7.))
        # rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
        # dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)

        # param_dict = {"alpha_h": alpha_h, "beta_h": beta_h, "h": h,"concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp, "dec_temp": dec_temp}
        # param_dict = {"concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp, "dec_temp": dec_temp}

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
            pbar.set_description("Mean ELBO  %6.2f, %6.2f" % (ar.tensor(loss[:5]).mean(), ar.tensor(loss[-20:]).mean()))
            alpha_h = pyro.param("alpha_h").data.numpy()
            beta_h = pyro.param("beta_h").data.numpy()
            print("alpha: ", alpha_h, " beta: ", beta_h)
            print('h: ', (alpha_h+beta_h)/alpha_h)        
            # print("alpha: ", alpha_h, " beta: ", beta_h)
            # print('dec: ', dec)
            # print('h': h)      
            # if ar.isnan(loss[-1]):
                # break

        self.loss = [l.cpu() for l in loss]
        
        # param_dict = {"alpha_h": alpha_h, "beta_h": beta_h, "h":h}
        
        # return self.loss, param_dict


    def analytical_posteriors(self):
        
        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.cpu().numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.cpu().numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.cpu().numpy()
        alpha_h = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_h = pyro.param("beta_lamb_r").data.cpu().numpy()
        concentration_dec_temp = pyro.param("concentration_dec_temp").data.cpu().numpy()
        rate_dec_temp = pyro.param("rate_dec_temp").data.cpu().numpy()
        
        param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
                      "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
                      "alpha_h": alpha_h, "beta_h": beta_h,
                      "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}
        
        x_lamb = np.arange(0.01,1.,0.01)
        
        y_lamb_pi = analytical_dists.Beta(x_lamb, alpha_lamb_pi, beta_lamb_pi)
        y_lamb_r = analytical_dists.Beta(x_lamb, alpha_lamb_r, beta_lamb_r)
        y_h = analytical_dists.Beta(x_lamb, alpha_h, beta_h)
        
        x_dec_temp = np.arange(0.01,10.,0.01)
        
        y_dec_temp = analytical_dists.Gamma(x_dec_temp, concentration=concentration_dec_temp, rate=rate_dec_temp)
        
        xs = [x_lamb, x_lamb, x_lamb, x_dec_temp]
        ys = [y_lamb_pi, y_lamb_r, y_h, y_dec_temp]
        
        return xs, ys, param_dict
    


    def plot_posteriors(self):
        
        #df, param_dict = self.sample_posteriors()
        
        xs, ys, param_dict = self.analytical_posteriors()
        
        # lamb_pi_name = "$\\lambda_{\\pi}$ as Beta($\\alpha$="+str(param_dict["alpha_lamb_pi"][0])+", $\\beta$="+str(param_dict["beta_lamb_pi"][0])+")"
        # lamb_r_name = "$\\lambda_{r}$ as Beta($\\alpha$="+str(param_dict["alpha_lamb_r"][0])+", $\\beta$="+str(param_dict["beta_lamb_r"][0])+")"
        h_name = "h"
        # dec_temp_name = "$\\gamma$ as Gamma(conc="+str(param_dict["concentration_dec_temp"][0])+", rate="+str(param_dict["rate_dec_temp"][0])+")"
        names = [h_name]
        xlabels = ["h",]
        #xlims = {"lamb_pi": [0,1], "lamb_r": [0,1], "dec_temp": [0,10]}
        
        for i in range(len(xs)):
            plt.figure()
            plt.title(names[i])
            plt.plot(xs[i],ys[i])
            plt.xlim([xs[i][0]-0.01,xs[i][-1]+0.01])
            plt.xlabel(xlabels[i])
            plt.show()
            
        print(param_dict)