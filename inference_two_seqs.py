
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
ar.set_default_dtype(ar.float32)

ar.set_num_threads(1)
class SingleInference(object):

    def __init__(self, agent, data, params):
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.params = params
        self.svi = None
        self.loss = []
        ar.manual_seed(5)

    def model(self):
        
        if self.params['infer_h']:
            alpha_h = ar.ones(1).to(device)
            beta_h = ar.ones(1).to(device)

            # sample initial vaue of parameter from Beta distribution
            h = pyro.sample('h', dist.Beta(alpha_h, beta_h))
        
        if self.params['infer_dec']:
            concentration_dec_temp = ar.tensor(1.).to(device)
            rate_dec_temp = ar.tensor(0.5).to(device)
            # sample initial vaue of parameter from normal distribution
            dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)

        if self.params['infer_both']:
            param_dict = {"h": h, "dec_temp":dec_temp}
        else:
            if self.params['infer_h']:
                param_dict = {"h":h}
            elif self.params['infer_dec']:
                param_dict = {"dec_temp":dec_temp}


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
                    if(tau==self.trials-1):
                        print(probs)
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.T), obs=curr_response)
                    


    def guide(self):



        if self.params['infer_h']:
            alpha_h = pyro.param("alpha_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
            beta_h = pyro.param("beta_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
            # sample initial vaue of parameter from Beta distribution
            h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)
        if self.params['infer_dec']:
            concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#interval(0., 7.))
            rate_dec_temp = pyro.param("rate_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
            dec_temp = pyro.sample('dec_temp', dist.Gamma(concentration_dec_temp, rate_dec_temp)).to(device)


        if self.params['infer_both']:
            param_dict = {"alpha_h": alpha_h, "beta_h": beta_h, "h": h,\
                          "concentration_dec_temp": concentration_dec_temp,\
                          "rate_dec_temp": rate_dec_temp, "dec_temp": dec_temp}
        else:
            if self.params['infer_h']:
                param_dict = {"alpha_h": alpha_h, "beta_h": beta_h, "h": h}
            elif self.params['infer_dec']:
                param_dict = {"concentration_dec_temp": concentration_dec_temp,\
                              "rate_dec_temp": rate_dec_temp, "dec_temp": dec_temp}

        self.param_dict = param_dict
        print(self.param_dict)
        return self.param_dict


    def init_svi(self, optim_kwargs={'lr': .01},
                 num_particles=10):
        
        pyro.clear_param_store()
    
        self.svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))


    def infer_posterior(self,
                        iter_steps=1000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}):

        """
        Perform SVI over free model parameters.
        """
 
        if self.svi is None:
            self.init_svi(optim_kwargs, num_particles)

        pbar = tqdm(range(iter_steps), position=0)
        for step in pbar:
            self.loss.append(ar.tensor(self.svi.step()).to(device))
            pbar.set_description("Mean ELBO  %6.2f, %6.2f" % (ar.tensor(self.loss[:5]).mean(), ar.tensor(self.loss[-20:]).mean()))
            
            if self.params['infer_h']:
                alpha_h = pyro.param("alpha_h").data.numpy()
                beta_h = pyro.param("beta_h").data.numpy()
                # print("alpha: ", alpha_h, " beta: ", beta_h)
                print('alpha_h: ', alpha_h, ', beta_h: ', beta_h) 
                print('h: ', (alpha_h+beta_h)/alpha_h)
            
            if self.params['infer_dec']:       
                alpha = pyro.param("concentration_dec_temp").data.numpy()
                beta = pyro.param("rate_dec_temp").data.numpy()
                dec = alpha/beta
                print('dec: ', dec)
      
            if ar.isnan(self.loss[-1]):
                break

        # self.loss += [l.cpu() for l in loss]
        
        return self.loss


    def analytical_posteriors(self):
        
        vals_dict = dict.fromkeys(tuple(self.param_dict.keys()))

        for key in vals_dict:
            try:
                vals_dict[key] = pyro.param(key).data.cpu().numpy()
            except:
                pass

        
        xs = []
        ys = []
        for key in vals_dict:
            if key == 'h':
                x_h = np.arange(0.01,1.,0.01)
                y_h = analytical_dists.Beta(x_h, vals_dict['alpha_h'], vals_dict['beta_h'])
                xs.append(x_h)
                ys.append(y_h)
            elif key == 'alpha_lamb_pi':
                x_lamb_pi = np.arange(0.01,1.,0.01)
                y_lamb_pi = analytical_dists.Beta(x_lamb_pi, vals_dict[key], vals_dict['beta_lamb_pi'])
                xs.append(x_lamb_pi)
                ys.append(y_lamb_pi)
            elif key == 'alpha_lamb_r':
                x_lamb_r = np.arange(0.01,1.,0.01)
                y_lamb_r = analytical_dists.Beta(x_lamb_r, vals_dict[key], vals_dict['beta_lamb_r'])
                xs.append(x_lamb_r)
                ys.append(y_lamb_r)
            elif key == 'rate_dec_temp':
                x_dec_temp = np.arange(0.01,10.,0.01)    
                y_dec_temp = analytical_dists.Gamma(x_dec_temp, concentration=vals_dict['concentration_dec_temp'],\
                                                    rate=vals_dict['rate_dec_temp'])
                xs.append(x_dec_temp)
                ys.append(y_dec_temp)

        return xs, ys, vals_dict
    


    def plot_posteriors(self):
        
        xs, ys, param_dict = self.analytical_posteriors()
        names = []
        xlabels = []
        for key in param_dict:
            if key == 'h':
                h_name = "h"
                names.append(h_name)
                xlabels.append("h")
            elif key == 'alpha_lamb_pi':
                lamb_pi_name = "$\\lambda_{\\pi}$ as Beta($\\alpha$="+str(param_dict["alpha_lamb_pi"][0])+", $\\beta$="+str(param_dict["beta_lamb_pi"][0])+")"
                names.append(lamb_pi_name)
                xlabels.append( "forgetting rate prior policies: $\\lambda_{\pi}$")
            elif key == 'alpha_lamb_r':
                lamb_r_name = "$\\lambda_{r}$ as Beta($\\alpha$="+str(param_dict["alpha_lamb_r"][0])+", $\\beta$="+str(param_dict["beta_lamb_r"][0])+")"
                names.append(lamb_r_name)
                xlabels.append("forgetting rate reward probabilities: $\\lambda_{r}$")
            elif key == 'rate_dec_temp':
                dec_temp_name = "$\\gamma$ as Gamma(conc="+str(param_dict["concentration_dec_temp"][0])+", rate="+str(param_dict["rate_dec_temp"][0])+")"
                "decision temperature: $\\gamma$"
                names.append(dec_temp_name)
                xlabels.append("decision temperature: $\\gamma$")

        #xlims = {"lamb_pi": [0,1], "lamb_r": [0,1], "dec_temp": [0,10]}
        
        for i in range(len(xs)):
            plt.figure()
            plt.title(names[i])
            plt.plot(xs[i],ys[i])
            plt.xlim([xs[i][0]-0.01,xs[i][-1]+0.01])
            plt.xlabel(xlabels[i])
            plt.show()
            
        print(param_dict)

    def save_parameters(self, fname):
        
        pyro.get_param_store().save(fname)
        
    def load_parameters(self, fname):
        
        pyro.get_param_store().load(fname)