
from logging import raiseExceptions
from pyroapi import optim
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
from torch.distributions import constraints, biject_to
import pyro
import pyro.distributions as dist
import agent as agt
import perception as prc
import action_selection as asl
import distributions as analytical_dists

device = ar.device("cpu")
ar.set_default_dtype(ar.float64)

ar.set_num_threads(1)


class GeneralGroupInference(object):

    def __init__(self, agent, data):

        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.nsubs = data[list(data.keys())[0]].shape[-1]
        self.svi = None
        self.loss = []
        self.npars = self.agent.perception.npars
        if agent.perception.mask is None:
            self.mask = ar.ones(data['actions'].shape) == 1

        
    def model(self):

        """
        Generative model of behavior with a NormalGamma
        prior over free model parameters.
        """

        # define hyper priors over model parameters
        a = pyro.param('a', ar.ones(self.npars), constraint=constraints.positive)
        lam = pyro.param('lam', ar.ones(self.npars), constraint=constraints.positive)
        tau = pyro.sample('tau', dist.Gamma(a, a/lam).to_event(1))

        sig = 1/ar.sqrt(tau)

        # each model parameter has a hyperprior defining group level mean
        m = pyro.param('m', ar.zeros(self.npars))
        s = pyro.param('s', ar.ones(self.npars), constraint=constraints.positive)
        mu = pyro.sample('mu', dist.Normal(m, s*sig).to_event(1))


        #for ind in range(self.nsubs):#pyro.plate("subject", len(self.data)):
        with pyro.plate('subject', self.nsubs) as ind:

            base_dist = dist.Normal(0., 1.).expand_by([self.npars]).to_event(1)
            transform = dist.transforms.AffineTransform(mu, sig)
            locs = pyro.sample('locs', dist.TransformedDistribution(base_dist, [transform]))

            self.agent.reset(locs)
            #self.agent.set_parameters(pol_lambda=lamb_pi, r_lambda=lamb_r, dec_temp=dec_temp)
            # print(self.agent.perception.alpha_0)
            # print(self.agent.perception.dirichlet_pol_params_init)

            # for tau in pyro.markov(range(self.trials)):
            for tau in pyro.markov(range(84)):
                for t in range(self.T):

                    if t==0:
                        prev_response = None
                    else:
                        prev_response = self.data["actions"][tau, t-1]

                    self.agent.perception.planets = self.data["planets"][tau]
                    observation = self.data["observations"][tau, t]
                    reward = self.data["rewards"][tau, t]
                    context = self.data['context_obs'][tau]

                    self.agent.update_beliefs(tau, t, observation, reward, prev_response, context)

                    if t < self.T-1:

                        probs = self.agent.perception.posterior_actions[-1]
                        #print(probs)
                        if ar.any(ar.isnan(probs)):
                            raise Exception('there was nans')

                        curr_response = self.data["actions"][tau, t]

                        pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.permute(1,2,0)), obs=curr_response)

    def guide(self):
        # approximate posterior. assume MF: each param has his own univariate Normal.

        trns = biject_to(constraints.positive)

        m_hyp = pyro.param('m_hyp', ar.zeros(2*self.npars))
        st_hyp = pyro.param('scale_tril_hyp',
                       ar.eye(2*self.npars),
                       constraint=constraints.lower_cholesky)

        hyp = pyro.sample('hyp',
                     dist.MultivariateNormal(m_hyp, scale_tril=st_hyp),
                     infer={'is_auxiliary': True})

        unc_mu = hyp[..., :self.npars]
        unc_tau = hyp[..., self.npars:]

        c_tau = trns(unc_tau)

        ld_tau = trns.inv.log_abs_det_jacobian(c_tau, unc_tau)
        ld_tau = dist.util.sum_rightmost(ld_tau, ld_tau.dim() - c_tau.dim() + 1)

        mu = pyro.sample("mu", dist.Delta(unc_mu, event_dim=1))
        tau = pyro.sample("tau", dist.Delta(c_tau, log_density=ld_tau, event_dim=1))

        m_locs = pyro.param('m_locs', ar.zeros(self.nsubs, self.npars))
        st_locs = pyro.param('scale_tril_locs',
                        ar.eye(self.npars).repeat(self.nsubs, 1, 1),
                        constraint=constraints.lower_cholesky)

        with pyro.plate('subject', self.nsubs):
                locs = pyro.sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))

        return {'tau': tau, 'mu': mu, 'locs': locs}


    def init_svi(self, optim_kwargs={'lr': .01},
                 num_particles=10):

        pyro.clear_param_store()

        self.svi = pyro.infer.SVI(model=self.model,
                  guide=self.guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))


        pyro.render_model(self.model, filename='model.pdf', render_params=True,render_distributions=True)
        pyro.render_model(self.guide, filename='guide.pdf', render_params=True ,render_distributions=True)
    def infer_posterior(self,
                        iter_steps=1000, optim_kwargs={'lr': .01},
                                     num_particles=10):
        """Perform SVI over free model parameters.
        """

        #pyro.clear_param_store()
        if self.svi is None:
            self.init_svi(optim_kwargs, num_particles)

        loss = []
        pbar = tqdm(range(iter_steps), position=0)

        for step in pbar:#range(iter_steps):
            loss.append(ar.tensor(self.svi.step()).to(device))
            pbar.set_description("Mean ELBO %6.2f" % ar.tensor(loss[-20:]).mean())
            if ar.isnan(loss[-1]):
                break

        self.loss += [l.cpu() for l in loss]

        # alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.numpy()
        # beta_lamb_pi = pyro.param("beta_lamb_pi").data.numpy()
        # alpha_lamb_r = pyro.param("alpha_lamb_r").data.numpy()
        # beta_lamb_r = pyro.param("beta_lamb_r").data.numpy()
        # alpha_h = pyro.param("alpha_lamb_r").data.numpy()
        # beta_h = pyro.param("beta_lamb_r").data.numpy()
        # concentration_dec_temp = pyro.param("concentration_dec_temp").data.numpy()
        # rate_dec_temp = pyro.param("rate_dec_temp").data.numpy()

        # param_dict = {"alpha_lamb_pi": alpha_lamb_pi, "beta_lamb_pi": beta_lamb_pi,
        #               "alpha_lamb_r": alpha_lamb_r, "beta_lamb_r": beta_lamb_r,
        #               "alpha_h": alpha_h, "beta_h": beta_h,
        #               "concentration_dec_temp": concentration_dec_temp, "rate_dec_temp": rate_dec_temp}

        # return self.loss#, param_dict


    def sample_posterior_predictive(self, n_samples=5):

        predictive = pyro.infer.Predictive(model=self.model, guide=self.guide, num_samples=n_samples)
        samples = predictive.get_samples()

        #pbar = tqdm(range(n_samples), position=0)

        # for n in pbar:
        #     pbar.set_description("Sample posterior depth")
        #     # get marginal posterior over planning depths
        #     post_samples = elbo.compute_marginals(self.model, config_enumerate(self.guide))
        #     print(post_samples)
        #     for name in post_samples.keys():
        #         post_sample_dict.setdefault(name, [])
        #         post_sample_dict[name].append(post_samples[name].probs.detach().clone())

        # for name in post_sample_dict.keys():
        #     post_sample_dict[name] = ar.stack(post_sample_dict[name]).numpy()

        # post_sample_df = pd.DataFrame(post_sample_dict)


        reordered_sample_dict = {}
        all_keys = []
        for key in samples.keys():
            if key[:3] != 'res':
                reordered_sample_dict[key] = np.array([])
                all_keys.append(key)

        reordered_sample_dict['subject'] = np.array([])

        #nsubs = len(self.data)
        for sub in range(self.nsubs):
            for key in set(all_keys):
                reordered_sample_dict[key] = np.append(reordered_sample_dict[key], samples[key][:,sub].detach().numpy())#.squeeze()
            reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]*n_samples).squeeze()

        # for key in samples.keys():
        #     if key[:3] != 'res':
        #         sub = int(key[-1])
        #         reordered_sample_dict[key[:-2]] = np.append(reordered_sample_dict[key[:-2]], samples[key].detach().numpy()).squeeze()
        #         reordered_sample_dict['subject'] = np.append(reordered_sample_dict['subject'], [sub]).squeeze()

        sample_df = pd.DataFrame(reordered_sample_dict)

        return sample_df


    def sample_posterior(self, n_samples=5):
        # keys = ["lamb_pi", "lamb_r", "h", "dec_temp"]

        param_names = self.agent.perception.param_names
        sample_dict = {param: [] for param in param_names}
        sample_dict["subject"] = []

        for i in range(n_samples):
            sample = self.guide()
            for key in sample.keys():
                sample.setdefault(key, ar.ones(1))

            par_sample = self.agent.locs_to_pars(sample["locs"])

            for param in param_names:
                sample_dict[param].extend(list(par_sample[param].detach().numpy()))

            sample_dict["subject"].extend(list(range(self.nsubs)))

        sample_df = pd.DataFrame(sample_dict)

        return sample_df


    def analytical_posteriors(self):

        alpha_lamb_pi = pyro.param("alpha_lamb_pi").data.cpu().numpy()
        beta_lamb_pi = pyro.param("beta_lamb_pi").data.cpu().numpy()
        alpha_lamb_r = pyro.param("alpha_lamb_r").data.cpu().numpy()
        beta_lamb_r = pyro.param("beta_lamb_r").data.cpu().numpy()
        alpha_h = pyro.param("alpha_h").data.cpu().numpy()
        beta_h = pyro.param("beta_h").data.cpu().numpy()
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


    def plot_posteriors(self, n_samples=5):

        #df, param_dict = self.sample_posteriors()

        #sample_df = self.sample_posterior_marginals(n_samples=n_samples)

        sample_df = self.sample_posterior(n_samples=n_samples)

        plt.figure()
        sns.displot(data=sample_df, x='h', hue='subject')
        plt.xlim([0,1])
        plt.show()

        plt.figure()
        sns.displot(data=sample_df, x='lamb_r', hue='subject')
        plt.xlim([0,1])
        plt.show()

        plt.figure()
        sns.displot(data=sample_df, x='lamb_pi', hue='subject')
        plt.xlim([0,1])
        plt.show()

        plt.figure()
        sns.displot(data=sample_df, x='dec_temp', hue='subject')
        plt.xlim([0,10])
        plt.show()

        # plt.figure()
        # sns.histplot(marginal_df["h_1"])
        # plt.show()

        return sample_df


    def save_parameters(self, fname):

        pyro.get_param_store().save(fname)


    def load_parameters(self, fname):

        pyro.get_param_store().load(fname)

class SingleInference(object):

    def __init__(self, agent, data, params):
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.params = params
        self.svi = None
        self.loss = []
        self.tol = 0.15
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
                self.agent.perception.planets = self.data["planets"][tau]
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

        if self.params['infer_h']:
            alpha_h = pyro.param("alpha_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
            beta_h = pyro.param("beta_h", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)#greater_than_eq(1.))
            h = pyro.sample('h', dist.Beta(alpha_h, beta_h)).to(device)

        if self.params['infer_dec']:
            concentration_dec_temp = pyro.param("concentration_dec_temp", ar.ones(1), constraint=ar.distributions.constraints.positive).to(device)
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
        # print(self.param_dict)
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
        print(optim_kwargs)
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
                # print('alpha_h: ', alpha_h, ', beta_h: ', beta_h) 
                # print('h: ', (alpha_h+beta_h)/alpha_h)
            
            if self.params['infer_dec']:       
                alpha = pyro.param("concentration_dec_temp").data.numpy()
                beta = pyro.param("rate_dec_temp").data.numpy()
                dec = alpha/beta
                # print('dec: ', dec)
      
            if ar.isnan(self.loss[-1]):
                break

        # self.loss += [l.cpu() for l in loss]
        
        return np.array(self.loss)

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
    


    def plot_posteriors(self,run_name = None,show=False):
        
        xs, ys, param_dict = self.analytical_posteriors()
        names = []
        xlabels = []
        inferred = []
        for key in param_dict:
            if key == 'h':
                h_name = "h"
                names.append(h_name)
                inferred.append('_h')
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
                inferred.append('_dec')
                xlabels.append("decision temperature: $\\gamma$")

        #xlims = {"lamb_pi": [0,1], "lamb_r": [0,1], "dec_temp": [0,10]}
        
        for i in range(len(xs)):
            fig = plt.figure()
            plt.title(names[i])
            plt.plot(xs[i],ys[i])
            plt.xlim([xs[i][0]-0.01,xs[i][-1]+0.01])
            plt.xlabel(xlabels[i])
            if show==True:
                plt.show()
            
            if run_name is not None:
                fig.savefig(run_name + inferred[i] + '.png', dpi=300)
        # print(param_dict)

    def save_parameters(self, fname):
        
        pyro.get_param_store().save(fname)
        # print(pyro.get_param_store()['h'])
        alpha_h = pyro.get_param_store()['alpha_h']
        beta_h = pyro.get_param_store()['beta_h']
        print((alpha_h + beta_h)/alpha_h)
    def load_parameters(self, fname):
        
        pyro.get_param_store().load(fname)
        # print(pyro.get_param_store()['alpha_h'],pyro.get_param_store()['beta_h'])

    def return_inferred_parameters(self):
        
        inferred_params = {}
        for key in self.param_dict:
            try:
                inferred_params[key] = pyro.param(key).data.numpy().tolist()
            except:
                pass

        return inferred_params


    def check_convergence(self, y, n=10):
        js = y.size - n + 1     # number of windows
        rmsd = []
        
        for j in range(js):
            x = y[j:j+n]
            mu_x = x.mean()
            rmsd.append(np.sqrt(np.sum((x - mu_x)**2)/(n-1)))
        
        if np.array(rmsd[-5:]).mean() <= self.tol:
            return True
        else :
            return False

class SingleInference_broken(object):

    def __init__(self, agent, data, params):
        self.agent = agent
        self.trials = agent.trials
        self.T = agent.T
        self.data = data
        self.params = params
        self.svi = None
        self.loss = []
        self.version = 'constrained'
        self.tol = 0.2
        ar.manual_seed(5)

    
    def __get_model(self):
        if self.version == 'constrained':
            return self.model_beta
        elif self.version == 'unconstrained':
            return self.model_log

    def __get_guide(self):
        if self.version == 'constrained':
            return self.guide_beta
        elif self.version == 'unconstrained':
            return self.guide_log


    def model_beta(self):
        
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
                    print(probs)
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
            

    def model_log(self):
        
        if self.params['infer_h']:
            mu_h = ar.ones(1).to(device)
            sigma_h = ar.ones(1).to(device)

            # sample initial vaue of parameter from Beta distribution
            h = pyro.sample('h', dist.Beta(mu_h, sigma_h))
        
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
                    # if(tau==self.trials-1):
                    #     print(probs)
                    pyro.sample('res_{}_{}'.format(tau, t), dist.Categorical(probs.T), obs=curr_response)
                    


    def guide_beta(self):

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
        # print(self.param_dict)
        return self.param_dict



    def guide_log(self):

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

        model = self.__get_model()
        guide = self.__get_guide()

        self.svi = pyro.infer.SVI(model=model,
                  guide=guide,
                  optim=pyro.optim.Adam(optim_kwargs),
                  loss=pyro.infer.Trace_ELBO(num_particles=num_particles,
                                  #set below to true once code is vectorized
                                  vectorize_particles=True))

    def check_convergence(self, y, n=10):
        js = y.size - n + 1     # number of windows
        rmsd = []
        
        for j in range(js):
            x = y[j:j+n]
            mu_x = x.mean()
            rmsd.append(np.sqrt(np.sum((x - mu_x)**2)/(n-1)))
        
        if np.array(rmsd[-5:]).mean() <= self.tol:
            return True
        else :
            return False


    def infer_posterior(self,
                        iter_steps=1000,
                        num_particles=10,
                        optim_kwargs={'lr': .01}):

        """
        Perform SVI over free model parameters.
        """
 
        if self.svi is None:
            self.init_svi(optim_kwargs, num_particles)

        # pbar = tqdm(range(iter_steps), position=0)
        for step in range(iter_steps):
            self.loss.append(ar.tensor(self.svi.step()).to(device))
            print(step)
            # pbar.set_description("Mean ELBO  %6.2f, %6.2f" % (ar.tensor(self.loss[:5]).mean(), ar.tensor(self.loss[-20:]).mean()))
            
            if self.params['infer_h']:
                alpha_h = pyro.param("alpha_h").data.numpy()
                beta_h = pyro.param("beta_h").data.numpy()
                # print("alpha: ", alpha_h, " beta: ", beta_h)
                # print('alpha_h: ', alpha_h, ', beta_h: ', beta_h) 
                # print('h: ', (alpha_h+beta_h)/alpha_h)
            
            if self.params['infer_dec']:       
                concentration_dec_temp = pyro.param("concentration_dec_temp").data.numpy()
                rate_dec_temp = pyro.param("rate_dec_temp").data.numpy()
                dec = concentration_dec_temp/rate_dec_temp
                # print('dec: ', dec)
      
            if ar.isnan(self.loss[-1]):
                break

        # self.loss += [l.cpu() for l in loss]
        
        return np.array(self.loss)

    def return_inferred_parameters(self):
        
        inferred_params = {}
        for key in self.param_dict:
            try:
                inferred_params[key] = pyro.param(key).data.numpy().tolist()
            except:
                pass

        return inferred_params

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
    


    def plot_posteriors(self,run_name = None,show=False):
        
        xs, ys, param_dict = self.analytical_posteriors()
        names = []
        xlabels = []
        inferred = []
        for key in param_dict:
            if key == 'h':
                h_name = "h"
                names.append(h_name)
                inferred.append('_h')
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
                inferred.append('_dec')
                xlabels.append("decision temperature: $\\gamma$")

        #xlims = {"lamb_pi": [0,1], "lamb_r": [0,1], "dec_temp": [0,10]}
        
        for i in range(len(xs)):
            fig = plt.figure()
            plt.title(names[i])
            plt.plot(xs[i],ys[i])
            plt.xlim([xs[i][0]-0.01,xs[i][-1]+0.01])
            plt.xlabel(xlabels[i])
            if show==True:
                plt.show()
            
            if run_name is not None:
                fig.savefig(run_name + inferred[i] + '.png', dpi=300)
        # print(param_dict)

    def save_parameters(self, fname):
        
        pyro.get_param_store().save(fname)
        
    def load_parameters(self, fname):
        
        pyro.get_param_store().load(fname)