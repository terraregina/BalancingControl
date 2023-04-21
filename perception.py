# from misc import ln, softmax
# import numpy as np
# import scipy.special as scs
# from misc import D_KL_nd_dirichlet, D_KL_dirichlet_categorical
# import torch as ar
# from sys import exit

from sim_parameters import arr_type
from misc import ln, softmax

import numpy as np
import scipy.special as scs
from misc import D_KL_nd_dirichlet, D_KL_dirichlet_categorical
from opt_einsum import contract

if arr_type == "numpy":
    import numpy as ar
    array = ar.array
else:
    import torch as ar
    array = ar.tensor
    device = ar.device("cuda") if ar.cuda.is_available() else ar.device("cpu")
    device = ar.device("cuda")
    device = ar.device("cpu")

    try:
        from inference_habit import device
    except:
        device = ar.device("cpu")
    ar.set_default_dtype(ar.float64)


class GroupFittingPerception(object):
    

    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 alpha_0 = None,
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 transition_matrix_context = None,
                 number_of_planets = None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=array([1]), dec_temp_cont = 1, nr=3,
                 possible_rewards=[-1,0,1], prior_context=None, npart=1, nsubs=1):
        
        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.generative_model_context = generative_model_context
        self.transition_matrix_context = transition_matrix_context
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.dec_temp_cont = dec_temp_cont
        self.policies = policies
        self.npi = policies.shape[0]
        self.actions = ar.unique(policies)
        self.npl = number_of_planets
        if transition_matrix_context is not None:
            self.nc = transition_matrix_context.shape[0]
        else:
            self.nc = 1
        self.na = len(self.actions)
        self.npart = npart
        self.nsubs = nsubs
        self.alpha_0 = alpha_0
        self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.nc,self.npart,self.nsubs)).to(device) + self.alpha_0[:,None, None, None]#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)
        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init\
             for k in range(self.npart)], dim=-1)\
                  for j in range(self.nsubs)], dim=-1)]        
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]
        
        #self.prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]
        if prior_context != None:
            self.prior_context = prior_context
        else:
            self.prior_context = array([1])
        #self.generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]
        self.observations = []
        self.rewards = []
        self.context_cues = []
        self.nr = nr
        self.reward_ind  = {}    
        for r, reward in enumerate(possible_rewards):
            self.reward_ind[reward] = r  
        self.possible_rewards = ar.tensor(possible_rewards)
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []
        self.curr_gen_mod_rewards = []
        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []
        self.posterior_contexts = []
        self.likelihoods = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t],:] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        self.big_trans_matrix = ar.moveaxis(self.big_trans_matrix, (0,1,2,3,4), (3,0,1,2,4))


    def locs_to_pars(self, locs):

        par_dict = {
                    "h": ar.sigmoid(locs[...,0]),
                    # "h": ar.exp(locs[...,0]+1),
                    # "pol_lambda": ar.sigmoid(locs[...,0]),
                    # "r_lambda": ar.sigmoid(locs[...,1]),
                    "dec_temp": 10*ar.sigmoid(locs[...,1])}
    
        return par_dict


    def reset(self):
    
        if len(self.dec_temp.shape) > 1:
            self.npart = self.dec_temp.shape[0]
            self.nsubs = self.dec_temp.shape[1]
        else:
            self.nsubs = self.dec_temp.shape[0]
            self.npart = 1
            #self.alpha_0 = self.alpha_0[None,:]
            # self.pol_lambda = self.pol_lambda[None,:]
            # self.r_lambda = self.r_lambda[None,:]
            self.dec_temp = self.dec_temp[None,:]
            self.alpha_0 = self.alpha_0[None,:]

        self.dirichlet_pol_params_init = ar.zeros((self.npi, self.nc, self.npart, self.nsubs)).to(device) + self.alpha_0.to(device)
        # self.dirichlet_pol_params_init = ar.zeros((self.npi, self.nc, self.npart)).to(device) + self.alpha_0[:,:,None]#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)

        self.dirichlet_rew_params = [ar.stack([ar.stack([self.dirichlet_rew_params_init\
             for k in range(self.npart)], dim=-1)\
                  for j in range(self.nsubs)], dim=-1)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]
        
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]
        
        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []
        self.likelihoods = []
        
        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []
        self.posterior_contexts = []
        self.curr_gen_mod_rewards = []


    def set_parameters(self, locs):

        par_dict = self.locs_to_pars(locs)

        if 'pol_lambda' in par_dict.keys():
            self.pol_lambda = par_dict['pol_lambda']
        if 'r_lambda' in par_dict.keys():
            self.r_lambda = par_dict['r_lambda']
        if 'dec_temp' in par_dict.keys():
            self.dec_temp = par_dict['dec_temp']
        if 'h' in par_dict.keys():
            self.alpha_0 = 1./par_dict['h']
            # self.alpha_0 = par_dict['h']

        elif 'alpha_0' in par_dict.keys():
            self.alpha_0 = par_dict['alpha_0']


    def make_current_messages(self, tau, t):
    
        generative_model_rewards = self.curr_gen_mod_rewards[-1]
        observations = ar.stack(self.observations[-t-1:])
        if len(observations.shape) == 1:
            observations = observations.reshape([observations.size()[0],1])
        obs_messages = []
        
        for n in range(self.nsubs):
            prev_obs = [self.generative_model_observations[o] for o in observations[-t-1:,n]]
            obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
            obs = [ar.stack(obs).T.to(device)]*self.nc
            # obs =  ar.stack(obs
            obs = [ar.stack(obs,dim=-1).to(device).to(device)]*self.npart
            obs_messages.append(ar.stack(obs,dim=-1).to(device))
            # prev_obs = [self.generative_model_observations[o] for o in observations[-t-1:,n]]
            # obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
            # obs = [ar.stack(obs).T.to(device)]*self.npart
            # obs_messages.append(ar.stack(obs, dim=-1))
        obs_messages = ar.stack(obs_messages, dim=-1).to(device)
        

        rew_messages = []
        for n in range(self.nsubs):
            rew_messages_participant = []
            for i in range(self.npart):
                curr_rew_mess =  []
                for ti in range(-t-1,0,1):
                    # [self.curr_gen_mod_rewards[ti][self.reward_ind[int(self.rewards[ti][n])],:,:,i,n].to(device) for ti in range(-t-1,0,1)]
                    rew_ind = self.reward_ind[int(self.rewards[ti][n])]
                    curr_rew_mess.append(self.curr_gen_mod_rewards[ti][rew_ind,:,:,i,n].to(device))
                
                rew_pref = [ar.matmul(ar.moveaxis(self.curr_gen_mod_rewards[-t-1][:,:,:,i,n],(0,1,2),(2,0,1)).to(device), self.prior_rewards).to(device)]*(self.T-t-1)        
                rew_messages_participant.append(ar.stack(curr_rew_mess + rew_pref, dim=-2))
            
            rew_messages_participant = ar.stack(rew_messages_participant,dim=-1)
            rew_messages.append(rew_messages_participant)
        rew_messages = ar.stack(rew_messages,dim=-1)

        if self.npl != None:
            rew_messages[:,0,:,:,:] = 1/self.possible_rewards.size()[0]
        self.obs_messages.append(obs_messages)
        self.rew_messages.append(rew_messages)


    def update_messages(self, tau, t):


        bwd = [ar.zeros((self.nh, self.npi, self.nc, self.npart, self.nsubs)).to(device)+1./self.nh]
        fwd = [ar.zeros((self.nh, self.npi, self.nc, self.npart, self.nsubs)).to(device)+self.prior_states[:,None,None,None,None]]


        fwd_norm = [ar.ones(self.npi, self.nc, self.npart, self.nsubs).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            tmp = ar.einsum('hpcnk,shpc,hcnk,hcnk->spcnk',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i+1,:],rew_messages[:,i+1,:]).to(device)

            bwd.append(tmp)
            norm = bwd[-1].sum(axis=0) + 1e-20

            bwd[-1] /= norm[None,:]
            
        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1,0,2,3,4,5).to(device)

 
        for i in range(self.T-1):
            tmp = ar.einsum('spcnk,shpc,scnk,scnk->hpcnk',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i,:],rew_messages[:,i,:]).to(device)
            fwd.append(tmp)

            norm = fwd[-1].sum(axis=0) + 1e-20
            fwd[-1] /= norm[None,:] 
            fwd_norm.append(norm.to(device))


        fwd_messages = ar.stack(fwd).permute(1,0,2,3,4,5).to(device)
                
        posterior = fwd_messages*bwd_messages*obs_messages[:,:,None,:,:]*rew_messages[:,:,None,:,:]
        norm = posterior.sum(axis = 0)
        fwd_norm.append(norm[-1])
        fwd_norms = ar.stack(fwd_norm).to(device)
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)


    def update_beliefs(self, tau, t, observation, reward, response, context=None):

        self.update_beliefs_states(tau, t,
                                    observation,
                                    reward)


        self.update_beliefs_policies(tau, t) 

        if tau == 0:
            prior_context = self.prior_context[:,None,None].repeat(1,self.npart,self.nsubs)

        else: #elif t == 0:
            prior_context = ar.einsum('ncpk,cpk-> npk',self.transition_matrix_context[:,:,None,None],self.posterior_contexts[-t-1])


        self.update_beliefs_context(tau, t, \
                                    reward, \
                                    # self.posterior_states[-1], \
                                    # self.posterior_policies[-1], \
                                    prior_context, \
                                    context=context)

        if t == self.T-1:
            self.update_beliefs_dirichlet_pol_params(tau, t)

        if t>0:
            self.update_beliefs_dirichlet_rew_params(tau, t, reward)


    def update_beliefs_states(self, tau, t, observation, reward):
    
        self.observations.append(observation)
        self.rewards.append(reward)
            
        self.update_messages(tau, t)


    def update_beliefs_policies(self, tau, t):

        likelihood = (self.fwd_norms[-1]+ 1e-20).prod(axis=0).to(device)
        norm = likelihood.sum(axis=0)
        likelihood = ar.pow(likelihood/norm[None,...],self.dec_temp[None,None,...]).to(device) #* ar.pow(norm,self.dec_temp)
        # likelihood /= likelihood.sum(axis=0)

        posterior= likelihood * self.prior_policies[-1]
        posterior /= posterior.sum(axis=0)
        self.posterior_policies.append(posterior)
        self.likelihoods.append(likelihood)


    def update_beliefs_context(self, tau, t, reward,\
                                # posterior_states,\
                                # posterior_policies,\
                                prior_context,\
                                context=None):

        # post_policies = (prior_context[ar.newaxis,:] * posterior_policies).sum(axis=1)
        post_policies = ar.einsum('pcnk, cnk -> pnk', self.posterior_policies[-1], prior_context).to(device)

        alpha = self.dirichlet_pol_params[-1]
        if t == self.T-1:
            chosen_pol = ar.argmax(post_policies,axis=0)

            alpha_prime = []
            for n in range(self.nsubs):
                part_spec = []
                for part in range(self.npart):
                    part_spec.append(ar.stack([prior_context[:,part,n] if p == int(chosen_pol[part,n]) else ar.zeros(self.nc) for p in range(self.npi)]))
                alpha_prime.append(ar.stack(part_spec,dim=-1))

            alpha_prime = self.dirichlet_pol_params[-1] + ar.stack(alpha_prime, dim=-1)

        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = ar.ones(1)
        else:
            if t>0:
                outcome_surprise = (self.posterior_policies[-1] * ln(self.fwd_norms[-1].prod(axis=0))).sum(axis=0)
                entropy = - (self.posterior_policies[-1] * ln(self.posterior_policies[-1])).sum(axis=0)
                policy_surprise = (self.posterior_policies[-1] * ar.digamma(alpha_prime)).sum(axis=0) - ar.digamma(alpha_prime.sum(axis=0))
            else:
                outcome_surprise = 0
                entropy = 0
                policy_surprise = 0
    
            if context is not None:
                context_obs_suprise = ar.stack([ln(self.generative_model_context[context].T+1e-10) for n in range(self.npart)],dim=-1).to(device)
                # context_obs_suprise = ar.stack([ln(self.generative_model_context[context[0]].T+1e-10) for n in range(self.npart)],dim=-1).to(device)
                # NOT GONNA WORK DURING FITTING!
                context_obs_suprise = ar.stack([context_obs_suprise for sub in range(self.nsubs)],dim=-1).to(device)
            else:
                context_obs_suprise = 0
            
            posterior = outcome_surprise + policy_surprise + entropy + context_obs_suprise           
            posterior = ar.nan_to_num(softmax(posterior+ln(prior_context)))


            posterior = ar.pow(posterior,self.dec_temp_cont)
            posterior /= posterior.sum(axis=0)
            self.posterior_contexts.append(posterior)

        if t<self.T-1:

            posterior_policies = ar.einsum('pcnk,cnk->pnk', self.posterior_policies[-1], self.posterior_contexts[-1])
            posterior_policies /= posterior_policies.sum(axis=0)

            posterior_actions = ar.zeros(self.na, self.npart,self.nsubs)
            for a in range(self.na):
                posterior_actions[a] = posterior_policies[self.policies[:,t] == a].sum(axis=0)
                
            self.posterior_actions.append(posterior_actions)
    

    def update_beliefs_dirichlet_pol_params(self, tau, t):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(self.posterior_policies[-1], axis=0).to(device)
        post_cont = self.posterior_contexts[-1]
        #print(chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        dirichlet_pol_params = (1-self.pol_lambda) * self.dirichlet_pol_params[-1] + (1 - (1-self.pol_lambda))*self.dirichlet_pol_params_init
        pols = ar.arange(self.npi).repeat((self.nc,1)).T

        mask = ar.stack([\
                ar.stack([pols == chosen_pol[:,p,n].repeat((self.npi,1)) for p in range(self.npart)],dim=-1)\
                for n in range(self.nsubs)\
            ],dim=-1)

        mask = mask*post_cont[None,:,:,:]
        dirichlet_pol_params +=  mask #posterior_context
        
        prior_policies = dirichlet_pol_params / dirichlet_pol_params.sum(axis=0)[None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        #prior_policies /= prior_policies.sum(axis=0)[None,:]
        
        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        #return dirichlet_pol_params, prior_policies


    def update_beliefs_dirichlet_rew_params(self, tau, t, reward):
        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,t,:,:,:] * posterior_policies[None,:,:,:]).sum(axis=1)
        st = ar.argmax(states, axis=0)
        # planets = ar.zeros([self.npl, self.nc])
        # planets[planets[st], ar.arange(self.nc)] = 1
        pl = self.planets.clone()
        planets = ar.arange(self.npl).repeat(self.npart,self.nsubs, 1).permute(2,0,1)


        # try:
        planets = ar.stack([planets == pl[st[c,:,:]].repeat((self.npl,1,1))\
            for c in range(self.nc)],dim=1)*self.posterior_contexts[-1][None,:,:,:]
        # except:
        #     planets = ar.stack([planets == ar.tensor(pl[st[c]]).repeat((self.npl,1))\
        #         for c in range(self.nc)],dim=1)*self.posterior_contexts[-1][None,:,:]

        # mask = ar.stack(\
        #     [planets if r == self.reward_ind[int(reward)] else ar.zeros(planets.shape) for r in range(self.nr)]\
        #     , dim=0)

        mask = []
        for r in range(self.nr):
            subject_specific = []
            for n in range(self.nsubs):
                rew = self.reward_ind[int(reward[n])]
                if rew == r:
                    subject_specific.append(planets[...,n])
                else:
                    subject_specific.append(ar.zeros(planets.shape[:-1]))
            mask.append(ar.stack(subject_specific, dim=-1))
        mask = ar.stack(mask,dim=0)

        dirichlet_rew_params = self.dirichlet_rew_params[-1] + mask
        
        generative_model_rewards = dirichlet_rew_params / dirichlet_rew_params.sum(axis=0)[None,...]
        self.dirichlet_rew_params.append(dirichlet_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))


class FittingPerception(object):

    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 alpha_0 = None,
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 transition_matrix_context = None,
                 number_of_planets = None,
                 T=5, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 dec_temp=array([1]), dec_temp_cont = 1, npl=3,nr=3,\
                 possible_rewards=[-1,0,1], prior_context=None, npart=1):
        
        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.generative_model_context = generative_model_context
        self.transition_matrix_context = transition_matrix_context
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.npl = npl
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.dec_temp_cont = dec_temp_cont
        self.policies = policies
        self.npi = policies.shape[0]
        self.actions = ar.unique(policies)
        self.npl = number_of_planets
        if transition_matrix_context is not None:
            self.nc = transition_matrix_context.shape[0]
        else:
            self.nc = 1
        self.na = len(self.actions)
        self.npart = npart
        self.alpha_0 = alpha_0
        self.dirichlet_rew_params_init = dirichlet_rew_params#ar.stack([dirichlet_rew_params]*self.npart, dim=-1)
        self.dirichlet_pol_params_init = ar.zeros((self.npi,self.nc,self.npart)).to(device) + self.alpha_0[:,None, None]#ar.stack([dirichlet_pol_params]*self.npart, dim=-1)
        self.dirichlet_rew_params = [ar.stack([self.dirichlet_rew_params_init]*self.npart, dim=-1).to(device)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]
        #self.prior_policies_init = self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]
        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]
        if prior_context != None:
            self.prior_context = prior_context
        else:
            self.prior_context = array([1])
            #self.generative_model_rewards_init = self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]
        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []
        self.context_cues = []
        self.nr = nr
        self.reward_ind  = {}    
        for r, reward in enumerate(possible_rewards):
            self.reward_ind[reward] = r  
        self.possible_rewards = ar.tensor(possible_rewards)
        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []
        self.curr_gen_mod_rewards = []
        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []
        self.posterior_contexts = []
        self.outcome_suprise = []
        self.policy_entropy = []
        self.context_obs_surprise = []
        self.policy_surprise = []
        self.likelihoods = []

        self.big_trans_matrix = ar.stack([ar.stack([generative_model_states[:,:,policies[pi,t],:] for pi in range(self.npi)]) for t in range(self.T-1)]).T.to(device)
        self.big_trans_matrix = ar.moveaxis(self.big_trans_matrix, (0,1,2,3,4), (3,0,1,2,4))


    def reset(self):
        self.npart = self.alpha_0.size()[0]
        self.dirichlet_pol_params_init = ar.zeros((self.npi, self.nc, self.npart)).to(device) + self.alpha_0.to(device)

        self.dirichlet_rew_params = [ar.stack([self.dirichlet_rew_params_init]*self.npart, dim=-1).to(device)]
        self.dirichlet_pol_params = [self.dirichlet_pol_params_init]

        self.prior_policies = [self.dirichlet_pol_params[0] / self.dirichlet_pol_params[0].sum(axis=0)[None,...]]

        self.generative_model_rewards = [self.dirichlet_rew_params[0] / self.dirichlet_rew_params[0].sum(axis=0)[None,...]]

        self.observations = []
        self.rewards = []

        #self.instantiate_messages()
        self.bwd_messages = []
        self.fwd_messages = []
        self.obs_messages = []
        self.rew_messages = []
        self.fwd_norms = []

        self.posterior_states = []
        self.posterior_policies = []
        self.posterior_actions = []
        self.likelihoods = []
        self.posterior_contexts = []
        self.outcome_suprise = []
        self.policy_entropy = []
        self.context_obs_surprise = []
        self.policy_surprise = []
        self.curr_gen_mod_rewards = []

        
    def make_current_messages(self, tau, t):

        if self.npl != None:
            generative_model_rewards = self.curr_gen_mod_rewards
        else:
            generative_model_rewards = self.generative_model_rewards


        prev_obs = [self.generative_model_observations[o] for o in self.observations[-t-1:]]
        obs = prev_obs + [ar.zeros((self.nh)).to(device)+1./self.nh]*(self.T-t-1)
        obs = [ar.stack(obs).T.to(device)]*self.nc
        obs = [ar.stack(obs,dim=-1).to(device).to(device)]*self.npart
        obs_messages = ar.stack(obs,dim=-1).to(device)

        rew_messages = ar.stack(\
            [
                ar.stack(
                    [generative_model_rewards[ti][self.reward_ind[int(self.rewards[ti])],:,:,i].to(device) for ti in range(-t-1,0,1)] \
                    + 
                    [ar.matmul(ar.moveaxis(generative_model_rewards[-t-1][:,:,:,i],(0,1,2),(2,0,1)).to(device), self.prior_rewards).to(device)]*(self.T-t-1)\
                ,dim=-2).to(device)\
            for i in range(self.npart)], dim=-1).to(device)

        if self.npl != None:
            rew_messages[:,0,:,:] = 1/self.possible_rewards.size()[0]

        self.obs_messages.append(obs_messages)
        self.rew_messages.append(rew_messages)


    def update_messages(self, tau, t):
        

        bwd = [ar.zeros((self.nh, self.npi, self.nc, self.npart)).to(device)+1./self.nh]
        fwd = [ar.zeros((self.nh, self.npi, self.nc, self.npart)).to(device)+self.prior_states[:,None,None,None]]

        fwd_norm = [ar.ones(self.npi, self.nc, self.npart).to(device)]

        self.make_current_messages(tau,t)

        obs_messages = self.obs_messages[-1]
        rew_messages = self.rew_messages[-1]

        for i in range(self.T-2,-1,-1):
            tmp = ar.einsum('hpcn,shpc,hcn,hcn->spcn',bwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i+1,:],rew_messages[:,i+1,:]).to(device)
            bwd.append(tmp)
            norm = bwd[-1].sum(axis=0) + 1e-20

            bwd[-1] /= norm[None,:]
            
        bwd.reverse()
        bwd_messages = ar.stack(bwd).permute(1,0,2,3,4).to(device)
 
        for i in range(self.T-1):
            tmp = ar.einsum('spcn,shpc,scn,scn->hpcn',fwd[-1],self.big_trans_matrix[...,i],obs_messages[:,i,:],rew_messages[:,i,:]).to(device)
            fwd.append(tmp)

            norm = fwd[-1].sum(axis=0) + 1e-20
            fwd[-1] /= norm[None,:] 
            fwd_norm.append(norm.to(device))


        fwd_messages = ar.stack(fwd).permute(1,0,2,3,4).to(device)
                
        posterior = fwd_messages*bwd_messages*obs_messages[:,:,None,:,:]*rew_messages[:,:,None,:,:]
        norm = posterior.sum(axis = 0)
        fwd_norm.append(norm[-1])
        fwd_norms = ar.stack(fwd_norm).to(device)
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]

        self.bwd_messages.append(bwd_messages)
        self.fwd_messages.append(fwd_messages)
        self.fwd_norms.append(fwd_norms)
        self.posterior_states.append(posterior)

        return posterior


    def update_beliefs(self, tau, t, observation, reward, response, context=None):

        self.update_beliefs_states(tau, t,
                                    observation,
                                    reward)


        self.update_beliefs_policies(tau, t) 

        if tau == 0:
            prior_context = self.prior_context[:,None].repeat(1,self.npart)

        else: #elif t == 0:
            prior_context = ar.einsum('ncp,cp-> np',self.transition_matrix_context[:,:,None],self.posterior_contexts[-t-1])


        self.update_beliefs_context(tau, t, \
                                    reward, \
                                    # self.posterior_states[-1], \
                                    # self.posterior_policies[-1], \
                                    prior_context, \
                                    context=context)

        if t == self.T-1:
            self.update_beliefs_dirichlet_pol_params(tau, t)

        if t>0:
            self.update_beliefs_dirichlet_rew_params(tau, t, reward)


    def update_beliefs_states(self, tau, t, observation, reward):
    
        self.observations.append(observation)
        self.rewards.append(reward)
            
        self.update_messages(tau, t)
        

    def update_beliefs_policies(self, tau, t):

        likelihood = (self.fwd_norms[-1]+ 1e-20).prod(axis=0).to(device)
        norm = likelihood.sum(axis=0)
        likelihood = ar.pow(likelihood/norm[None,:,:],self.dec_temp[None,None,:]).to(device) #* ar.pow(norm,self.dec_temp)


        posterior= likelihood * self.prior_policies[-1]
        posterior /= posterior.sum(axis=0)
        self.posterior_policies.append(posterior)
        self.likelihoods.append(likelihood)
   

    def update_beliefs_context(self, tau, t, reward,\
                                # posterior_states,\
                                # posterior_policies,\
                                prior_context,\
                                context=None):

        # post_policies = (prior_context[ar.newaxis,:] * posterior_policies).sum(axis=1)
        post_policies = ar.einsum('pcn, cn -> pn', self.posterior_policies[-1], prior_context).to(device)

        alpha = self.dirichlet_pol_params[-1]
        if t == self.T-1:

            chosen_pol = ar.argmax(post_policies,axis=0)
            alpha_prime = self.dirichlet_pol_params[-1]

            alph_prim = []
            for part in range(self.npart):
                alph_prim.append(
                    ar.stack([prior_context[:,part] if p == int(chosen_pol[part]) else ar.zeros(self.nc) for p in range(self.npi)],dim=0)
                )
            alpha_prime = alpha_prime + ar.stack(alph_prim, dim=-1)
            
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = ar.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[ar.newaxis,:]).sum(axis=1)[:,ar.newaxis] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            if t>0:
                outcome_surprise = (self.posterior_policies[-1] * ln(self.fwd_norms[-1].prod(axis=0))).sum(axis=0)
                entropy = - (self.posterior_policies[-1] * ln(self.posterior_policies[-1])).sum(axis=0)
                policy_surprise = (self.posterior_policies[-1] * ar.digamma(alpha_prime)).sum(axis=0) - ar.digamma(alpha_prime.sum(axis=0))
            else:
                outcome_surprise = 0
                entropy = 0
                policy_surprise = 0
    
            if context is not None:
                context_obs_suprise = ar.stack([ln(self.generative_model_context[context]+1e-10) for n in range(self.npart)],dim=-1).to(device)
            else:
                context_obs_suprise = 0
            
            posterior = outcome_surprise + policy_surprise + entropy + context_obs_suprise           
            posterior = ar.nan_to_num(softmax(posterior+ln(prior_context)))
            

            posterior = ar.pow(posterior,self.dec_temp_cont)
            posterior /= posterior.sum(axis=0)

            self.posterior_contexts.append(posterior)
            # self.context_obs_surprise.append(context_obs_suprise)
            # self.outcome_suprise.append(outcome_surprise)
            # self.policy_entropy.append(entropy)
            # self.policy_surprise.append(policy_surprise)

        if t<self.T-1:

            posterior_policies = ar.einsum('pcn,cn->pn', self.posterior_policies[-1], self.posterior_contexts[-1])
            posterior_policies /= posterior_policies.sum(axis=0)
            

            posterior_actions = ar.zeros(self.na, self.npart).to(device)
            for a in range(self.na):
                posterior_actions[a] = posterior_policies[self.policies[:,t] == a].sum(axis=0)
                
            self.posterior_actions.append(posterior_actions)
    

    def update_beliefs_dirichlet_pol_params(self, tau, t):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(self.posterior_policies[-1], axis=0).to(device)
        post_cont = self.posterior_contexts[-1]
        #print(chosen_pol)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        dirichlet_pol_params = (1-self.pol_lambda) * self.dirichlet_pol_params[-1] + (1 - (1-self.pol_lambda))*self.dirichlet_pol_params_init
        pols = ar.arange(self.npi).repeat((self.nc,1)).T
        mask = ar.stack([pols == chosen_pol[:,p].repeat((self.npi,1)) for p in range(self.npart)],dim=-1)*post_cont[None,:,:]
        dirichlet_pol_params +=  mask #posterior_context
        
        prior_policies = dirichlet_pol_params / dirichlet_pol_params.sum(axis=0)[None,...]#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
        #prior_policies /= prior_policies.sum(axis=0)[None,:]
        
        self.dirichlet_pol_params.append(dirichlet_pol_params.to(device))
        self.prior_policies.append(prior_policies.to(device))

        #return dirichlet_pol_params, prior_policies


    def update_beliefs_dirichlet_rew_params(self, tau, t, reward):

        posterior_states = self.posterior_states[-1]
        posterior_policies = self.posterior_policies[-1]
        states = (posterior_states[:,t,:,:] * posterior_policies[None,:,:]).sum(axis=1)
        st = ar.argmax(states, axis=0)
        

        if self.npl != None:
            pl = self.planets
            nh = self.npl
        else:
            pl = ar.arange(self.nh)
            nh = self.nh
        
        states_mask = ar.arange(nh).repeat(self.npart,1).T
        
        states_mask = ar.stack([states_mask == pl[st[c]].repeat((nh,1))\
            for c in range(self.nc)],dim=1)*self.posterior_contexts[-1][None,:,:]

        mask = ar.stack(\
            [states_mask if r == self.reward_ind[int(reward)] else ar.zeros(states_mask.shape) for r in range(self.nr)]\
            , dim=0)

        dirichlet_rew_params = self.dirichlet_rew_params[-1] + mask
        
        generative_model_rewards = dirichlet_rew_params / dirichlet_rew_params.sum(axis=0)[None,...]
        self.dirichlet_rew_params.append(dirichlet_rew_params.to(device))
        self.generative_model_rewards.append(generative_model_rewards.to(device))


class HierarchicalPerception(object):

    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 policies,
                 dirichlet_pol_params = None,
                 dirichlet_rew_params = None,
                 generative_model_context = None,
                 T=4, trials=10, pol_lambda=0, r_lambda=0, non_decaying=0,
                 possible_rewards = None,
                 dec_temp_cont = 1,
                 dec_temp=1., init_planets = None,):

        self.generative_model_observations = generative_model_observations
        if len(generative_model_states.shape)<4:
            self.generative_model_states = generative_model_states[:,:,:,None]
        else:
            self.generative_model_states = generative_model_states

        self.transition_matrix_context = transition_matrix_context
        self.prior_rewards = prior_rewards
        self.prior_states = prior_states
        dims = list(prior_policies.shape)
        dims = [trials] + dims
        self.prior_policies = ar.zeros(dims)
        self.prior_policies[:] = prior_policies[None,...]
        self.npi = prior_policies.shape[0]
        self.T = T
        self.trials = trials
        self.nh = prior_states.shape[0]
        self.pol_lambda = pol_lambda
        self.r_lambda = r_lambda
        self.non_decaying = non_decaying
        self.dec_temp = dec_temp
        self.planets = init_planets
        self.dec_temp_cont = dec_temp_cont
        self.possible_rewards = possible_rewards
        self.policies = policies
        if possible_rewards is not None:
            self.reward_ind = {}
            for r, reward in enumerate(possible_rewards):
                self.reward_ind[reward] = r  

        if len(generative_model_rewards.shape) > 2:
            dims = list(generative_model_rewards.shape)
            self.generative_model_rewards =  ar.zeros([self.trials, self.T] + dims)
            self.infer_context = True
            self.nc = generative_model_rewards.shape[2]
            self.nr = dims[0]
            self.generative_model_rewards[:] = generative_model_rewards[None,None,:,:,:]
        else:
            self.nc = 1
            dims = list(generative_model_rewards.shape)
            self.generative_model_rewards =  ar.zeros([self.trials, self.T] + dims + [1])
            self.generative_model_rewards[:] = self.generative_model_rewards[None,None,:,:,None]

        if possible_rewards is not None:
            self.npl = generative_model_rewards.shape[-2]
        

        if dirichlet_pol_params is not None:
            dims = list(dirichlet_pol_params.shape)
            dims = [self.trials] + dims
            self.dirichlet_pol_params = ar.zeros(dims)
            self.dirichlet_pol_params[:] = dirichlet_pol_params[None,None,...]

        if dirichlet_rew_params is not None:
            dims = list(dirichlet_rew_params.shape)
            dims = [self.trials, self.T] + dims
            self.dirichlet_rew_params = ar.zeros(dims)
            self.dirichlet_rew_params[:] = dirichlet_rew_params[None,None,...]

        if generative_model_context is not None:
            self.generative_model_context = generative_model_context
            # for c in range(self.nc):
            #     for state in range(self.npl):
            #         self.generative_model_rewards[:,state,c] = \
            #             self.dirichlet_rew_params[:,state,c] / self.dirichlet_rew_params[:,planet_type,c].sum()
            # for c in range(self.nc):
            #     for state in range(self.npl):
            #         self.generative_model_rewards[:,:,:,state,c] = (self.dirichlet_rew_params[0,0,state,c] / self.dirichlet_rew_params[0,0,state,c].sum())[None,None,...]
        if self.npi is not None:
            self.initiate_planet_rewards(self.generative_model_rewards[0,0,...])

        self.instantiate_messages()
        print('decision_temp context', self.dec_temp_cont)
    
    def reset(self):

        self.generative_model_rewards[:] = self.generative_model_rewards[0,0].clone().detach()[None,None,...]
        self.dirichlet_rew_params[:] = self.dirichlet_rew_params[0,0].clone().detach()[None,None,...]
        self.dirichlet_pol_params = ar.ones_like(self.dirichlet_pol_params)
        self.prior_policies = ar.ones_like(self.prior_policies) / self.npi

        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,:,:,state,c] = (self.dirichlet_rew_params[0,0,:,state,c] / self.dirichlet_rew_params[0,0,:,state,c].sum())[None,None,...]

        ### check if it works?
        if self.npi is not None:
            self.initiate_planet_rewards(self.generative_model_rewards[0,0,...])
        self.instantiate_messages()


    def initiate_planet_rewards(self,generative_model_rewards):#,tau):
            

        self.current_gen_model_rewards = ar.zeros([self.nr, self.nh, self.nc])

        for p in range(self.nh):
            self.current_gen_model_rewards[:,p,:] = generative_model_rewards[:,self.planets[p],:]
        

    def instantiate_messages(self):
        self.bwd_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.npi, self.nc))
        self.bwd_messages[:,:,:,-1,:, :] = 1./self.nh
        self.fwd_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.npi, self.nc))
        self.fwd_messages[:,:,:, 0, :, :] = self.prior_states[None,None,:, None, None]
        self.fwd_norms = ar.zeros((self.trials, self.T, self.T+1, self.npi, self.nc))
        self.fwd_norms[:,:,0,:,:] = 1.

        self.obs_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = ar.tile(self.obs_messages,(self.T,1)).T

        self.rew_messages = ar.zeros((self.trials, self.T, self.nh, self.T, self.nc))
        #self.rew_messages[:] = ar.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T

        for c in range(self.nc):

            if self.npi != None:
                curr_gen_model_rewards = self.current_gen_model_rewards[:,:,c]
            else:
                curr_gen_model_rewards = self.generative_model_rewards[0,0,:,:,c]

            self.rew_messages[:,:,:,:,c] = self.prior_rewards.dot(curr_gen_model_rewards)[None, None,:,None]

            for pi, cstates in enumerate(self.policies):
                for t, u in enumerate(ar.flip(cstates, [0])):#axis = 0)):
                    tp = self.T - 2 - t
                    self.bwd_messages[0,0,:,tp,pi,c] = (self.bwd_messages[0,0,:,tp+1,pi,c]*\
                                                self.obs_messages[0,0,:, tp+1,c]*\
                                                self.rew_messages[0,0,:, tp+1,c])[None,None,...]
                    bwd_message = (self.bwd_messages[0,0,:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u,c]))
                    bwd_message /= bwd_message.sum()
                    self.bwd_messages[:,:,:,tp, pi,c] = bwd_message[None,None,...]


    def update_messages(self, tau, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(ar.flip(cs[:], [0])):#axis = 0)):
                self.bwd_messages[tau,t,:,self.T-2-i,pi,c] = self.bwd_messages[tau,t,:,self.T-1-i,pi,c]*\
                                                self.obs_messages[tau,t,:,self.T-1-i,c]*\
                                                self.rew_messages[tau,t,:, self.T-1-i,c]
                self.bwd_messages[tau,t,:,self.T-2-i,pi,c] = self.bwd_messages[tau,t,:,self.T-2-i,pi,c]\
                    .dot(self.generative_model_states[:,:,u,c])

                norm = self.bwd_messages[tau,t,:,self.T-2-i,pi,c].sum()
                if norm > 0:
                    self.bwd_messages[tau,t,:,self.T-2-i, pi,c] /= norm

        if len(cs[:]) > 0:
            for i, u in enumerate(cs[:]):
                self.fwd_messages[tau,t,:, 1+i, pi,c] = (self.fwd_messages[tau,t,:,i, pi,c]*\
                                                self.obs_messages[tau,t,:, i,c]*\
                                                self.rew_messages[tau,t,:, i,c])

                self.fwd_messages[tau,t,:, 1+i, pi,c] = \
                                                (self.generative_model_states[:,:,u,c].\
                                                dot(self.fwd_messages[tau,t,:, 1+i, pi,c]))
                # self.fwd_norms[tau,t,:t,pi,c] = self.fwd_norms[tau,t-1,:t,pi,c]                                                
                self.fwd_norms[tau,t,1+i,pi,c] = self.fwd_messages[tau,t,:,1+i,pi,c].sum()
                if self.fwd_norms[tau,t,1+i, pi,c] > 0: #???? Shouldn't this not happen?
                    self.fwd_messages[tau,t,:,1+i, pi,c] /= self.fwd_norms[tau,t,1+i,pi,c]
        # print('a')


    def reset_preferences(self, t, new_preference, policies):

        self.prior_rewards = new_preference.copy()

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,ar.newaxis]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(ar.flip(cstates, axis = 0)):
                    tp = self.T - 2 - i
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u,c])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()


    def update_beliefs_states(self, tau, t, observation, reward, possible_policies):
        # estimate expected state distribution
        # if t == 2:
        #     print('breakpoint')
            # self.instantiate_messages(policies)

        if t>0:
            self.obs_messages[tau,t,:,:t,:] = self.obs_messages[tau,t-1,:,:t,:]
            self.rew_messages[tau,t,:,:t,:] = self.rew_messages[tau,t-1,:,:t,:]


        self.obs_messages[tau,t,:,t,:] = self.generative_model_observations[observation][:,None]
        
        if self.npi is not None:
            if t==0:
                for c in range(self.nc):
                    self.rew_messages[tau,:,:,:,c] = self.prior_rewards.dot(self.current_gen_model_rewards[:,:,c])[None,:,None]

                self.rew_messages[tau,t,:,t,:] = 1/self.current_gen_model_rewards.shape[0]  # same as 1 / number of rewards
            elif t>0:
                ind = self.reward_ind[reward]
                self.rew_messages[tau,t,:,t,:] = self.current_gen_model_rewards[ind]
                self.bwd_messages[tau,t,...] = self.bwd_messages[tau,t-1,...]
        else:
            self.rew_messages[tau,t,:,t,:] = self.generative_model_rewards[tau,t,reward]

        for c in range(self.nc):
            for pi, cs in enumerate(self.policies):
                if self.prior_policies[tau,pi,c] > 1e-15 and pi in possible_policies:
                    self.update_messages(tau, t, pi, cs, c)
                else:
                    self.fwd_messages[tau,t,:,:,pi,c] = 0#1./self.nh
                    if t!=0:
                        self.fwd_norms[tau,t,:t+1,pi,c] = self.fwd_norms[tau,t-1,:t+1,pi,c]
                        
        #estimate posterior state distribution
        posterior = self.fwd_messages[tau,t]*self.bwd_messages[tau,t]*self.obs_messages[tau,t,:,:,None,:]*self.rew_messages[tau,t,:,:,None,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[tau,t,-1] = norm[-1]
        non_zero = norm > 0
        posterior[:,non_zero] /= norm[non_zero]
        return posterior
        # return ar.nan_to_num(posterior)


    def update_beliefs_policies(self, tau, t):

        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms[tau,t].prod(axis=0)
        posterior = ar.power(likelihood,self.dec_temp) * self.prior_policies[tau] / \
            (ar.power(likelihood,self.dec_temp) * self.prior_policies[tau]).sum(axis=0)[None,:]
        #likelihood /= likelihood.sum(axis=0)[None,:]
        #posterior/= posterior.sum(axis=0)[None,:]
        #posterior = ar.nan_to_num(posterior)

        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        return posterior, likelihood


    def update_beliefs_context(self, tau, t, reward,\
                               posterior_states,\
                                posterior_policies,\
                                prior_context,\
                                policies,\
                                context=None):


        post_policies = (prior_context[None,:] * posterior_policies).sum(axis=1)

        alpha = self.dirichlet_pol_params[tau].copy()
        if t == self.T-1:
            chosen_pol = ar.argmax(post_policies)
            inf_context = ar.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params[tau].copy()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = ar.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[ar.newaxis,:]).sum(axis=1)[:,ar.newaxis] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            if t>0:
                # the higher the reward for a given policy and a given context the lower the logarithm
                outcome_surprise = (posterior_policies * ln(self.fwd_norms[tau,t].prod(axis=0))).sum(axis=0)
                # positive quantity
                entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
                #policy_surprise = (post_policies[:,ar.newaxis] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
                # 
                policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            else:
                outcome_surprise = 0
                entropy = 0
                policy_surprise = 0
                
            if context is not None:
                context_obs_suprise = ln(self.generative_model_context[context]+1e-10)
            else:
                context_obs_suprise = 0

            posterior = outcome_surprise + policy_surprise + entropy + context_obs_suprise
            posterior = ar.nan_to_num(softmax(posterior+ln(prior_context)))

            posterior = ar.power(posterior,self.dec_temp_cont)
            posterior /= posterior.sum()

            return [posterior, outcome_surprise, entropy, policy_surprise, context_obs_suprise]


    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(posterior_policies, axis=0)
        if tau < self.trials-1:
            self.dirichlet_pol_params[tau+1] = (1-self.pol_lambda) * self.dirichlet_pol_params[tau] + 1 - (1-self.pol_lambda)
            self.dirichlet_pol_params[tau+1,chosen_pol,:] += posterior_context
            self.prior_policies[tau+1] = self.dirichlet_pol_params[tau+1].copy()#ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[None,:])
            self.prior_policies[tau+1] /= self.prior_policies[tau+1].sum(axis=0)[None,:]
            # if tau < self.trials-1:
            #     self.dirichlet_pol_params[tau+1] = self.dirichlet_pol_params[tau]
            
            # print(tau,t, posterior_context)

            return self.dirichlet_pol_params[tau+1], self.prior_policies[tau+1]

        else:
            return self.dirichlet_pol_params[tau], self.prior_policies[tau]


        # else:
        #     return self.dirichlet_pol_params[tau], self.prior_policies[tau]


    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):

        dirichlet_rew_params = self.dirichlet_rew_params[tau,t,:,:,:].copy()
        dirichlet_rew_params = (1-self.r_lambda) * dirichlet_rew_params + 1 - (1-self.r_lambda)
        states = (posterior_states[:,t,:,:] * posterior_policies[ar.newaxis,:,:]).sum(axis=1)
        st = ar.argmax(states, axis=0)
        planets = ar.zeros([self.npl, self.nc])
        planets[self.planets[st], ar.arange(self.nc)] = 1
        dirichlet_rew_params[self.reward_ind[reward],:,:] += planets * posterior_context[None,:]
        generative_model_rewards = dirichlet_rew_params.copy()
        rew_messages = ar.zeros_like(self.rew_messages[tau,t].copy())

        for c in range(self.nc):
            if self.npl is not None:
                for state in range(self.npl):
                    generative_model_rewards[:,state,c] /= generative_model_rewards[:,state,c].sum()
                    rew_messages[:,:,c] = self.prior_rewards.dot(generative_model_rewards[:,:,c])[self.planets][:,None]
            else:
                for state in range(self.nh):
                    generative_model_rewards[:,state,c] /= generative_model_rewards[:,state,c].sum()
                    rew_messages[:,:,c] = self.prior_rewards.dot(generative_model_rewards[:,:,c])[:,None]
    

        if t<self.T-1:
            self.dirichlet_rew_params[tau,t+1] = dirichlet_rew_params
            self.generative_model_rewards[tau,t+1] = generative_model_rewards
            # self.rew_messages[tau,t+1:,:,t+1:,:] = rew_messages[None,:,t+1:,:]
        elif tau<self.trials-1:
            self.dirichlet_rew_params[tau+1,0] = dirichlet_rew_params
            self.generative_model_rewards[tau+1,0] = generative_model_rewards
            # self.rew_messages[tau+1,0:,:,0:,:] = rew_messages[None,:,0:,:]

        if not self.npi is None and t == self.T-1 and tau < self.trials-1:
            self.dirichlet_rew_params[tau+1,1:] = dirichlet_rew_params[None,:]
            self.generative_model_rewards[tau+1,1:] = generative_model_rewards[None,:]
        
        # self.initiate_planet_rewards(generative_model_rewards)
        # else:
        #     if t<self.T-1:
        #         self.dirichlet_rew_params[tau,t+1] = dirichlet_rew_params
        #         self.generative_model_rewards[tau,t+1] = generative_model_rewards
        #         self.rew_messages[tau,t+1:,:,t+1:,:] = rew_messages[None,:,t+1:,:]
        #         if self.npi is not None:
        #             self.initiate_planet_rewards(self.generative_model_rewards[tau,t+1])
        #     elif tau<self.trials-1:
        #         self.dirichlet_rew_params[tau+1,0] = dirichlet_rew_params
        #         self.generative_model_rewards[tau+1,0] = generative_model_rewards
        #         self.rew_messages[tau+1,0:,:,0:,:] = rew_messages[None,:,0:,:]


        return dirichlet_rew_params
        

class TwoStepPerception(object):
    def __init__(self,
                 generative_model_observations,
                 generative_model_states,
                 generative_model_rewards,
                 transition_matrix_context,
                 prior_states,
                 prior_rewards,
                 prior_policies,
                 dirichlet_pol_params = None,
                 dirichlet_rew_params = None,
                 T=5):

        self.generative_model_observations = generative_model_observations.copy()
        self.generative_model_states = generative_model_states.copy()
        self.generative_model_rewards = generative_model_rewards.copy()
        self.transition_matrix_context = transition_matrix_context.copy()
        self.prior_rewards = prior_rewards.copy()
        self.prior_states = prior_states.copy()
        self.prior_policies = prior_policies.copy()
        self.T = T
        self.nh = prior_states.shape[0]
        if len(generative_model_rewards.shape) > 2:
            self.infer_context = True
            self.nc = generative_model_rewards.shape[2]
        else:
            self.nc = 1
            self.generative_model_rewards = self.generative_model_rewards[:,:,ar.newaxis]
        if dirichlet_pol_params is not None:
            self.dirichlet_pol_params = dirichlet_pol_params.copy()
        if dirichlet_rew_params is not None:
            self.dirichlet_rew_params = dirichlet_rew_params.copy()


        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                       -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()


    def instantiate_messages(self, policies):
        npi = policies.shape[0]

        self.bwd_messages = ar.zeros((self.nh, self.T, npi, self.nc))
        self.bwd_messages[:,-1,:, :] = 1./self.nh
        self.fwd_messages = ar.zeros((self.nh, self.T, npi, self.nc))
        self.fwd_messages[:, 0, :, :] = self.prior_states[:, ar.newaxis, ar.newaxis]

        self.fwd_norms = ar.zeros((self.T+1, npi, self.nc))
        self.fwd_norms[0,:,:] = 1.

        self.obs_messages = ar.zeros((self.nh, self.T, self.nc)) + 1/self.nh#self.prior_observations.dot(self.generative_model_observations)
        #self.obs_messages = ar.tile(self.obs_messages,(self.T,1)).T

        self.rew_messages = ar.zeros((self.nh, self.T, self.nc))
        #self.rew_messages[:] = ar.tile(self.prior_rewards.dot(self.generative_model_rewards),(self.T,1)).T

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,ar.newaxis]
            for pi, cstates in enumerate(policies):
                for t, u in enumerate(ar.flip(cstates, axis = 0)):
                    tp = self.T - 2 - t
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_messages(self, t, pi, cs, c=0):
        if t > 0:
            for i, u in enumerate(ar.flip(cs[:t], axis = 0)):
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-i,pi,c]*\
                                                self.obs_messages[:,t-i,c]*\
                                                self.rew_messages[:, t-i,c]
                self.bwd_messages[:,t-1-i,pi,c] = self.bwd_messages[:,t-1-i,pi,c]\
                    .dot(self.generative_model_states[:,:,u])

                norm = self.bwd_messages[:,t-1-i,pi,c].sum()
                if norm > 0:
                    self.bwd_messages[:,t-1-i, pi,c] /= norm

        if len(cs[t:]) > 0:
           for i, u in enumerate(cs[t:]):
               self.fwd_messages[:, t+1+i, pi,c] = self.fwd_messages[:,t+i, pi,c]*\
                                                self.obs_messages[:, t+i,c]*\
                                                self.rew_messages[:, t+i,c]
               self.fwd_messages[:, t+1+i, pi,c] = \
                                                self.generative_model_states[:,:,u].\
                                                dot(self.fwd_messages[:, t+1+i, pi,c])
               self.fwd_norms[t+1+i,pi,c] = self.fwd_messages[:,t+1+i,pi,c].sum()
               if self.fwd_norms[t+1+i, pi,c] > 0: #???? Shouldn't this not happen?
                   self.fwd_messages[:,t+1+i, pi,c] /= self.fwd_norms[t+1+i,pi,c]

    def reset_preferences(self, t, new_preference, policies):

        self.prior_rewards = new_preference.copy()

        for c in range(self.nc):
            self.rew_messages[:,:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,ar.newaxis]
            for pi, cstates in enumerate(policies[t:]):
                for i, u in enumerate(ar.flip(cstates, axis = 0)):
                    tp = self.T - 2 - i
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp+1,pi,c]*\
                                                self.obs_messages[:, tp+1,c]*\
                                                self.rew_messages[:, tp+1,c]
                    self.bwd_messages[:,tp,pi,c] = self.bwd_messages[:,tp,pi,c]\
                        .dot(self.generative_model_states[:,:,u])
                    self.bwd_messages[:,tp, pi,c] /= self.bwd_messages[:,tp,pi,c].sum()

    def update_beliefs_states(self, tau, t, observation, reward, policies, possible_policies):
        #estimate expected state distribution
        if t == 0:
            self.instantiate_messages(policies)

        self.obs_messages[:,t,:] = self.generative_model_observations[observation][:,ar.newaxis]

        self.rew_messages[:,t,:] = self.generative_model_rewards[reward]

        for c in range(self.nc):
            for pi, cs in enumerate(policies):
                if self.prior_policies[pi,c] > 1e-15 and pi in possible_policies:
                    self.update_messages(t, pi, cs, c)
                else:
                    self.fwd_messages[:,:,pi,c] = 0#1./self.nh #0

        #estimate posterior state distribution
        posterior = self.fwd_messages*self.bwd_messages*self.obs_messages[:,:,ar.newaxis,:]*self.rew_messages[:,:,ar.newaxis,:]
        norm = posterior.sum(axis = 0)
        self.fwd_norms[-1] = norm[-1]
        posterior /= norm
        return ar.nan_to_num(posterior)

    def update_beliefs_policies(self, tau, t, gamma=4):

        #print((prior_policies>1e-4).sum())
        likelihood = self.fwd_norms.prod(axis=0)
        posterior = ar.power(likelihood,gamma) * self.prior_policies
        posterior/= posterior.sum(axis=0)[ar.newaxis,:]
        #posterior = softmax(ln(self.fwd_norms).sum(axis = 0)+ln(self.prior_policies))

        #ar.testing.assert_allclose(post, posterior)

        return posterior, likelihood


    def update_beliefs_context(self, tau, t, reward, posterior_states, posterior_policies, prior_context, policies):

        post_policies = (prior_context[ar.newaxis,:] * posterior_policies).sum(axis=1)
        beta = self.dirichlet_rew_params.copy()
        states = (posterior_states[:,t,:] * post_policies[ar.newaxis,:,ar.newaxis]).sum(axis=1)
        beta_prime = self.dirichlet_rew_params.copy()
        beta_prime[reward] = beta[reward] + states

#        for c in range(self.nc):
#            for state in range(self.nh):
#                self.generative_model_rewards[:,state,c] =\
#                ar.exp(scs.digamma(beta_prime[:,state,c])\
#                       -scs.digamma(beta_prime[:,state,c].sum()))
#                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()
#
#            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,ar.newaxis]
#
#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0

        alpha = self.dirichlet_pol_params.copy()
        if t == self.T-1:
            chosen_pol = ar.argmax(post_policies)
            inf_context = ar.argmax(prior_context)
            alpha_prime = self.dirichlet_pol_params.copy()
            alpha_prime[chosen_pol,:] += prior_context
            #alpha_prime[chosen_pol,inf_context] = self.dirichlet_pol_params[chosen_pol,inf_context] + 1
        else:
            alpha_prime = alpha


        if self.nc == 1:
            posterior = ar.ones(1)
        else:
            # todo: recalc
            #outcome_surprise = ((states * prior_context[ar.newaxis,:]).sum(axis=1)[:,ar.newaxis] * (scs.digamma(beta_prime[reward]) - scs.digamma(beta_prime.sum(axis=0)))).sum(axis=0)
            outcome_surprise = (posterior_policies * ar.log(self.fwd_norms.prod(axis=0))).sum(axis=0)
            entropy = - (posterior_policies * ln(posterior_policies)).sum(axis=0)
            #policy_surprise = (post_policies[:,ar.newaxis] * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            policy_surprise = (posterior_policies * scs.digamma(alpha_prime)).sum(axis=0) - scs.digamma(alpha_prime.sum(axis=0))
            posterior = outcome_surprise + policy_surprise + entropy

                        #+ ar.nan_to_num((posterior_policies * ln(self.fwd_norms).sum(axis = 0))).sum(axis=0)#\

#            if tau in range(90,120) and t == 1:
#                #print(tau, ar.exp(outcome_surprise), ar.exp(policy_surprise))
#                print(tau, ar.exp(outcome_surprise[1])/ar.exp(outcome_surprise[0]), ar.exp(policy_surprise[1])/ar.exp(policy_surprise[0]))


            posterior = ar.nan_to_num(softmax(posterior+ln(prior_context)))

        return posterior


    def update_beliefs_dirichlet_pol_params(self, tau, t, posterior_policies, posterior_context = [1]):
        assert(t == self.T-1)
        chosen_pol = ar.argmax(posterior_policies, axis=0)
#        self.dirichlet_pol_params[chosen_pol,:] += posterior_context.sum(axis=0)/posterior_context.sum()
        alpha = 0.3#0.3
        self.dirichlet_pol_params = (1-alpha) * self.dirichlet_pol_params + 1 - (1-alpha)
        self.dirichlet_pol_params[chosen_pol,:] += posterior_context
        self.prior_policies[:] = ar.exp(scs.digamma(self.dirichlet_pol_params) - scs.digamma(self.dirichlet_pol_params.sum(axis=0))[ar.newaxis,:])
        self.prior_policies /= self.prior_policies.sum(axis=0)

        return self.dirichlet_pol_params

    def update_beliefs_dirichlet_rew_params(self, tau, t, reward, posterior_states, posterior_policies, posterior_context = [1]):
        states = (posterior_states[:,t,:,:] * posterior_policies[ar.newaxis,:,:]).sum(axis=1)
        state = ar.argmax(states)
        old = self.dirichlet_rew_params.copy()
#        self.dirichlet_rew_params[:,state,:] = (1-0.4) * self.dirichlet_rew_params[:,state,:] #+1 - (1-0.4)
#        self.dirichlet_rew_params[reward,state,:] += 1#states * posterior_context[ar.newaxis,:]
        alpha = 0.6#0.3#1#0.3#0.05
        self.dirichlet_rew_params[:,3:,:] = (1-alpha) * self.dirichlet_rew_params[:,3:,:] +1 - (1-alpha)
        self.dirichlet_rew_params[reward,:,:] += states * posterior_context[ar.newaxis,:]
        for c in range(self.nc):
            for state in range(self.nh):
                self.generative_model_rewards[:,state,c] =\
                ar.exp(scs.digamma(self.dirichlet_rew_params[:,state,c])\
                       -scs.digamma(self.dirichlet_rew_params[:,state,c].sum()))
                self.generative_model_rewards[:,state,c] /= self.generative_model_rewards[:,state,c].sum()

            self.rew_messages[:,t+1:,c] = self.prior_rewards.dot(self.generative_model_rewards[:,:,c])[:,ar.newaxis]

#        for c in range(self.nc):
#            for pi, cs in enumerate(policies):
#                if self.prior_policies[pi,c] > 1e-15:
#                    self.update_messages(t, pi, cs, c)
#                else:
#                    self.fwd_messages[:,:,pi,c] = 1./self.nh #0

        return self.dirichlet_rew_params

# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNXK0Okxxdddddddddxkk0KXWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNX0kdoc:;,'''''''''''''''''',;coxOXWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMMMMWX0xoc;,''''''''''''''''''''''''''''''';lx0WMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMMMWKxl:,''''''''''''''''''''''''''''''''''''''',cxXWMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMW0xc,'''''''''''''''''''',:l:'''''cdc,''''''''''''';dKWMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMXkl,''''''''''''''''''''''';kWKdooodKMO;''''''''''''''';dKMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMXx:''''''''''''''''''''''''''c0NXOkkkk0XKd;';;''''''''''''':kNMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMNk:''''''''''''''''''''''',okxkX0l;''''',:dKK0XO:''''''''''''',lKMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMKl,'''''''''''''''''''''''',oONWk;''''''''''cKNkc;''''''''''''''':0WMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMWO:'''''''''''''''''''''''''''',ONo''''''''''',kNl'''''''''''''''''';kWMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMWO;'''''''''''''''''''''''''''',c0Wx,'''''''''':0WOo:''''''''''''''''';kWMMMMMMMMMMMMMMMMM
# MMMMMMMMMM0:''''''''''''''''''',d00d,''''oXNXXx:''''''',l0X000l'''''''''''''''''';OWMMMMMMMMMMMMMMMM
# MMMMMMMMMXl'''''''''''',::''''':0MMO;'''';ol:o0KOddoodx0XOc,,,,''''''''''''''''''':0MMMMMMMMMMMMMMMM
# MMMMMMMMMk,''''''''''':kNXxloxOKWMMNKOdc:dK0d,lXMKkkkxOWNo'''''''''''''''''''''''''lXMMMMMMMMMMMMMMM
# MMMMMMMMNo'''''''''''';dKWMWNKkdooodx0NNNWWXx;:xkc'',coxdccdc'''''''''''''''''''''',xWMMMMMMMMMMMMMM
# MMMMMMMMK:''''''''''''';kWWOl;''''''',:dXMNd'''''''',kN0kOXWk;'''''''''''''''''''''';OMMMMMMMMMMMMMM
# MMMMMMMMO;'''''''''''',xWNd,''''''''''''c0WKc''';lllxXN0xxkKNKkOo,'''''''''''''''''''dWMMMMMMMMMMMMM
# MMMMMMMMO,'''''''',lxdkNMk,''''''''''''''lXW0ooloOKXMXo,''';xNWk:,'''''''''''''''''';OMMMMMMMMMMMMMM
# MMMMMMMMO;'''''''';OMMMMWd''''''''''''''':0MMMMNo,cOMO;'''''cXWk;,'''''''''''''''''';0MMMMMMMMMMMMMM
# MMMMMMMM0;'''''''',:lldXMk;''''''''''''''lXW0dddcdOKWNkc;,;l0WX0x;'''''''''''''''''':0MMMMMMMMMMMMMM
# MMMMMMMMKc'''''''''''',dWNx,''''''''''''c0MKc''',ldookNNXKXWNk:,,'''''''''''''''''''oNMMMMMMMMMMMMMM
# MMMMMMMMNd''''''''''''':0MW0l;''''''',:xXMXl''''''''';OXkook0l''''''''''''''''''''',xWMMMMMMMMMMMMMM
# MMMMMMMMMO;'''''''''''c0WMWNNKOxdoodx0NWWMNOl,''''''',::,'',;,'''''''''''''''''''''';kNMMMMMMMMMMMMM
# MMMMMMMMMXl'''''''''''ckXOl:oxOXWMMN0kdllONKl,''''''''''''''''''''''''''''''''''''''',oKMMMMMMMMMMMM
# MMMMMMMMMWk,''''''''''',;,'''''oNMWd,'''',:;''''''''''''''''''''''''''''''''''''''''''':kNMMMMMMMMMM
# MMMMMMMMMMXl'''''''''''''''''''ck0kc'''''''''''''''''''''''''''''''''''''''''''''''''''',oKMMMMMMMMM
# MMMMMMMMMMMO;''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''':kNMMMMMMM
# MMMMMMMMMMMWd,''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''',lKWMMMMM
# MMMMMMMMMMMMXl''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''';kWMMMM
# MMMMMMMMMMMMMKc'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''oNMMMM
# MMMMMMMMMMMMMM0c'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''',:lox0NMMMMM
# MMMMMMMMMMMMMMMKl''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''',dKNWMMMMMMMMM
# MMMMMMMMMMMMMMMMXo,''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''',xWMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMNd,'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''cKMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMWk;'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''';OMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMWk;''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''l0NMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMWx,'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''dNMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMXl''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''';xNMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMNo''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''':0MMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMNo''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''',kWMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMMXl''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''oNMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMM0:''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''';kWMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMWx,'''''''''''''''''''''''''''''''''''''''''''''''''''''''''',;o0WMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMMKc'''''''''''''''''''''''''''''''''''''''''',:oxkkkxxddoooddk0XWMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMMWx,''''''''''''''''''''''''''''''''''''''''';xNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMM0:''''''''''''''''''''''''''''''''''''''''',xWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMMNo'''''''''''''''''''''''''''''''''''''''''':KMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMMWx,''''''''''''''''''''''''''''''''''''''''''lNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMMWx;'''''''''''''''''''''''''''''''''''''''''''oNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMMWx,''''''''''''''''''''''''''''''''''''''''''''dWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMMNx,'''''''''''''''''''''''''''''''''''''''''''''dWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMMNd,''''''''''''''''''''''''''''''''''''''''''''''xWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMMNd,'''''''''''''''''''''''''''''''''''''''''''''',kMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
# MMMMMMMMMMMWk,''''''''''''''''''''''''''''''''''''''''''''''';OMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM