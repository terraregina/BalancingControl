"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""

import numpy as np
from perception import HierarchicalPerception
from misc import ln, softmax
import scipy.special as scs
import torch as ar


try:
    from inference_two_seqs import device
except:
    device = ar.device("cpu")


class FittingAgent(object):

    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None,
                 prior_context = None,
                 learn_habit = False,
                 learn_rew = False,
                 trials = 1, T = 10, number_of_states = 6,
                 number_of_rewards = 2,
                 number_of_policies = 10):

        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection

        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        self.nr = number_of_rewards

        self.T = T
        self.trials = trials

        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = ar.eye(self.npi, dtype = int).to(device)

        self.possible_polcies = self.policies.clone().detach()

        self.actions = ar.unique(self.policies).to(device)
        self.na = len(self.actions)

        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = ar.ones(self.nh).to(device)
            self.prior_states /= self.prior_states.sum()

        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = ar.ones(1).to(device)
            self.nc = 1

        if prior_policies is not None:
            self.prior_policies = prior_policies[:,None]#ar.tile(prior_policies, (1,self.nc)).T
        else:
            self.prior_policies = ar.ones((self.npi)).to(device)/self.npi

        self.learn_habit = learn_habit
        self.learn_rew = learn_rew

        #set various data structures
        self.actions = ar.zeros((trials, T), dtype = int).to(device)
        self.observations = ar.zeros((trials, T), dtype = int).to(device)
        self.rewards = ar.zeros((trials, T), dtype = int).to(device)
        self.posterior_actions = ar.zeros((trials, T-1, self.na)).to(device)
        self.posterior_rewards = ar.zeros((trials, T, self.nr)).to(device)
        self.posterior_contexts = ar.zeros((trials, T, self.nc)).to(device)

        self.control_probs  = ar.zeros((trials, T, self.na)).to(device)
        self.log_probability = 0
        if hasattr(self.perception, 'generative_model_context'):
            self.context_obs = ar.zeros(trials, dtype=int).to(device)


    def reset(self, param_dict):

        self.actions = ar.zeros((self.trials, self.T), dtype = int).to(device)
        self.observations = ar.zeros((self.trials, self.T), dtype = int).to(device)
        self.rewards = ar.zeros((self.trials, self.T), dtype = int).to(device)
        self.posterior_actions = ar.zeros((self.trials, self.T-1, self.na)).to(device)
        self.posterior_rewards = ar.zeros((self.trials, self.T, self.nr)).to(device)
        self.control_probs  = ar.zeros((self.trials, self.T, self.na)).to(device)
        self.log_probability = 0
        if hasattr(self.perception, 'generative_model_context'):
            self.context_obs = ar.zeros(self.trials, dtype=int).to(device)

        self.set_parameters(**param_dict)
        self.perception.reset()


    def initiate_planet_rewards(self):
        
        gen_mod_rewards = np.zeros([self.nr, self.nh, self.nc])
        for p in range(self.nh):
            gen_mod_rewards[:,p,:] =\
            self.perception.generative_model_rewards[-1][:,self.planets[p],:]
            self.percetion.curr_gen_model_rewards = gen_mod_rewards
        
        # return gen_mod_rewards

    def update_beliefs(self, tau, t, observation, reward, response, context=None):
        

        if t == 0:
            self.initiate_planet_rewards()
            
        #print(observation)
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
        if context is not None:
            self.context_obs[tau] = context

        if t == 0:
            self.possible_polcies = ar.arange(0,self.npi,1, dtype=ar.long).to(device)
        else:
            #TODO!
            # wow so inefficient. probably rather remove for fitting...
            possible_policies = self.policies[:,t-1]==response
            prev_pols = ar.zeros(self.npi, dtype=bool).to(device)
            prev_pols[:] = False
            prev_pols[self.possible_polcies] = True
            new_pols = own_logical_and(possible_policies, prev_pols).to(device)
            self.possible_polcies = ar.where(new_pols==True)[0].to(device)
            
            # TODO once 1D intersect exists
            #self.possible_polcies = ar.intersect1d(self.possible_polcies, possible_policies)
           #self.log_probability += ln(self.posterior_actions[tau,t-1,response])

        self.perception.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward,
                                         #self.policies,
                                         self.possible_polcies)

        #update beliefs about policies
        self.perception.update_beliefs_policies(tau, t) #self.posterior_policies[tau, t], self.likelihood[tau,t]
        if tau == 0:
            prior_context = self.prior_context
        else: #elif t == 0:
            prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
        #    else:
        #        prior_context = ar.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])

        print(tau,t)
        print("prior", prior_context)
        print("post", self.posterior_context[tau, t])

        # if t < self.T-1:
        #     #post_pol = ar.matmul(self.posterior_policies[tau, t], self.posterior_context[tau, t])
        #     self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t)

        if self.nc>1 and t>=0:
            
            if hasattr(self, 'context_obs'): 
                c_obs = self.context_obs[tau]
            else:
                c_obs = None

        self.perception.update_beliefs_context(tau, t, \
                                                reward, \
                                                self.posterior_states[tau, t], \
                                                self.posterior_policies[tau, t], \
                                                prior_context, \
                                                self.policies,\
                                                context=c_obs)

        if t == self.T-1 and self.learn_habit:
            self.perception.update_beliefs_dirichlet_pol_params(tau, t)


        if False:
            self.posterior_rewards[tau, t-1] = ar.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t])
        #if reward > 0:
        # check later if stuff still works!
        if self.learn_rew:# and t>0:#==self.T-1:
            self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward)



class BayesianPlanner(object):

    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None,
                 prior_context = None,
                 learn_habit = False,
                 learn_rew = False,
                 trials = 1, T = 4, number_of_states = 6,
                 number_of_rewards = 3,
                 number_of_policies = 8, number_of_planets = None):

        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection

        #set parameters of the agent
        self.npl = number_of_planets
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        self.nr = number_of_rewards
        
        if prior_context is not None:
            self.nc = prior_context.size
        
        self.T = T

        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = np.eye(self.npi, dtype = int)

        self.possible_polcies = self.policies.copy()

        self.actions = np.unique(self.policies)
        self.na = len(self.actions)

        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = np.ones(self.nh)
            self.prior_states /= self.prior_states.sum()

        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = np.ones(1)
            self.nc = 1

        if prior_policies is not None:
            self.prior_policies = np.tile(prior_policies, (1,self.nc)).T
        else:
            self.prior_policies = np.ones((self.npi,self.nc))/self.npi

        self.learn_habit = learn_habit
        self.learn_rew = learn_rew

        #set various data structures
        self.actions = np.zeros((trials, T), dtype = int)
        self.posterior_states = np.zeros((trials, T, self.nh, T, self.npi, self.nc))
        self.posterior_policies = np.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = np.zeros((trials, self.npi, self.nc))
        if not number_of_planets is None:
            self.posterior_dirichlet_rew = np.zeros((trials, T, self.nr, self.npl, self.nc))
        else:
            self.posterior_dirichlet_rew = np.zeros((trials, T, self.nr, self.nh, self.nc))

        self.observations = np.zeros((trials, T), dtype = int)
        self.rewards = np.zeros((trials, T), dtype = int)
        self.posterior_context = np.ones((trials, T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[np.newaxis,np.newaxis,:]
        self.likelihood = np.zeros((trials, T, self.npi, self.nc))
        self.prior_policies = np.zeros((trials, self.npi, self.nc))
        self.prior_policies[:] = prior_policies[np.newaxis,:,:]
        self.posterior_actions = np.zeros((trials, T-1, self.na))
        self.posterior_rewards = np.zeros((trials, T, self.nr))
        self.log_probability = 0
        if hasattr(self.perception, 'generative_model_context'):
            self.context_obs = np.zeros(trials, dtype=int)
        self.prior_actions = np.zeros([trials, T-1, np.unique(self.policies).size])
        self.outcome_suprise = np.zeros([trials, T, self.nc])
        self.policy_entropy = np.zeros([trials, T, self.nc])
        self.context_obs_surprise = np.zeros([trials, T, self.nc])
        
    def reset(self, params, fixed):

        self.actions[:] = 0
        self.posterior_states[:] = 0
        self.posterior_policies[:] = 0
        self.posterior_dirichlet_pol[:] = 0
        self.posterior_dirichlet_rew[:] =0
        self.observations[:] = 0
        self.rewards[:] = 0
        self.posterior_context[:,:,:] = self.prior_context[np.newaxis,np.newaxis,:]
        self.likelihood[:] = 0
        self.posterior_actions[:] = 0
        self.posterior_rewards[:] = 0
        self.log_probability = 0
        self.perception.reset(params, fixed)

    # code set up to look at unique planet types
    #this function creates a reward matrix for the given planet constelation
    def initiate_planet_rewards(self):
        
        gen_mod_rewards = np.zeros([self.nr, self.nh, self.nc])
        for p in range(self.nh):
            gen_mod_rewards[:,p,:] =\
            self.perception.generative_model_rewards[:,self.planets[p],:]
        
        return gen_mod_rewards

    def update_beliefs(self, tau, t, observation, reward, response, context=None):
        
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward
        self.perception.planets = self.planets
        if context is not None:
            self.context_obs[tau] = context

        if t == 0:
            self.possible_polcies = np.arange(0,self.npi,1).astype(np.int32)
        else:
            possible_policies = np.where(self.policies[:,t-1]==response)[0]
            self.possible_polcies = np.intersect1d(self.possible_polcies, possible_policies)
            self.log_probability += ln(self.posterior_actions[tau,t-1,response])

        self.perception.current_gen_model_rewards = self.initiate_planet_rewards()
        self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward,
                                         self.policies,
                                         self.possible_polcies)

        #!#print("start", self.world.environment.starting_position[tau],"\n")
        #!#print("posterior over states for two contexts")
        #!#print(self.posterior_states[tau,t,:,t,:,0].T)
        #!#print("\n")
        #!#print(self.posterior_states[tau,t,:,t,:,1].T)
        #update beliefs about policies
        self.posterior_policies[tau, t], self.likelihood[tau,t] = self.perception.update_beliefs_policies(tau, t)
        

        #!#print("\nself.posterior_policies[tau, t]\n", self.posterior_policies[tau, t])
        #!#print("planet conf", self.world.environment.planet_conf[tau])
        #!#print("c1: ", self.policies[np.argmax(self.posterior_policies[tau, t][:,0])])
        #!#print("c2: ", self.policies[np.argmax(self.posterior_policies[tau, t][:,1])])

        if tau == 0:
            prior_context = self.prior_context
        else: #elif t == 0:
            prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))

#            else:
#                prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])


        if self.nc>1 and t>=0:
            
            if hasattr(self, 'context_obs'): 
                c_obs = self.context_obs[tau]
            else:
                c_obs = None

            self.posterior_context[tau, t,:], self.outcome_suprise[tau, t,:],\
            self.policy_entropy[tau,t,:], self.context_obs_surprise[tau,t,:] = self.perception.update_beliefs_context(tau, t, \
                                                reward, \
                                                self.posterior_states[tau, t], \
                                                self.posterior_policies[tau, t], \
                                                prior_context, \
                                                self.policies,\
                                                context=c_obs)
        else:
            self.posterior_context[tau,t] = 1
        
        if t < self.T-1:
            post_pol = np.dot(self.posterior_policies[tau, t], self.posterior_context[tau, t])
            self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t, post_pol)

        if t == self.T-1 and self.learn_habit:
            self.posterior_dirichlet_pol[tau], self.prior_policies[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
                                                            self.posterior_policies[tau,t], \
                                                            self.posterior_context[tau,t])

        # if False:
        #     self.posterior_rewards[tau, t-1] = np.einsum('rsc,spc,pc,c->r',
        #                                           self.perception.generative_model_rewards,
        #                                           self.posterior_states[tau,t,:,t],
        #                                           self.posterior_policies[tau,t],
        #                                           self.posterior_context[tau,t])
        #if reward > 0:
        if self.learn_rew and t>0:#==self.T-1:
            if hasattr(self,  'trial_type'):
                if not self.trial_type[tau] == 2:
                    self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                            self.posterior_states[tau, t], \
                                                            self.posterior_policies[tau, t], \
                                                            self.posterior_context[tau,t])
            else:
                raise('Agent not extinguishing reward!')

    def generate_response(self, tau, t):

        #get response probability
        posterior_states = self.posterior_states[tau, t]
        posterior_policies = np.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, t])
        posterior_policies /= posterior_policies.sum()
        
        # average likelihood refers to averaging over contexts
        avg_likelihood = np.einsum('pc,c->p', self.likelihood[tau,t], self.posterior_context[tau, t])
        avg_likelihood /= avg_likelihood.sum()
        prior = np.einsum('pc,c->p', self.prior_policies[tau-1], self.posterior_context[tau, t])
        prior /= prior.sum()
        self.prior_actions = self.estimate_action_probability(tau,t,prior)
        #print(self.posterior_context[tau, t])
        non_zero = posterior_policies > 0
        controls = self.policies[:, t]#[non_zero]
        actions = np.unique(controls)
        # posterior_policies = posterior_policies[non_zero]
        # avg_likelihood = avg_likelihood[non_zero]
        # prior = prior[non_zero]

        self.actions[tau, t] = self.action_selection.select_desired_action(tau,
                                        t, posterior_policies, controls, avg_likelihood, prior)


        return self.actions[tau, t]


    def estimate_action_probability(self, tau, t, posterior_policies):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[self.policies[:,t] == a].sum()


        return control_prob


class BayesianPlanner_old(object):

    def __init__(self, perception, action_selection, policies,
                 prior_states = None, prior_policies = None,
                 prior_context = None,
                 learn_habit = False,
                 learn_rew = False,
                 trials = 1, T = 10, number_of_states = 6,
                 number_of_rewards = 2,
                 number_of_policies = 10):

        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection

        #set parameters of the agent
        self.nh = number_of_states #number of states
        self.npi = number_of_policies #number of policies
        self.nr = number_of_rewards

        self.T = T

        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = np.eye(self.npi, dtype = int)

        self.possible_polcies = self.policies.copy()

        self.actions = np.unique(self.policies)
        self.na = len(self.actions)

        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = np.ones(self.nh)
            self.prior_states /= self.prior_states.sum()

        if prior_context is not None:
            self.prior_context = prior_context
            self.nc = prior_context.shape[0]
        else:
            self.prior_context = np.ones(1)
            self.nc = 1

        if prior_policies is not None:
            self.prior_policies = np.tile(prior_policies, (1,self.nc)).T
        else:
            self.prior_policies = np.ones((self.npi,self.nc))/self.npi

        self.learn_habit = learn_habit
        self.learn_rew = learn_rew

        #set various data structures
        self.actions = np.zeros((trials, T), dtype = int)
        self.posterior_states = np.zeros((trials, T, self.nh, T, self.npi, self.nc))
        self.posterior_policies = np.zeros((trials, T, self.npi, self.nc))
        self.posterior_dirichlet_pol = np.zeros((trials, self.npi, self.nc))
        self.posterior_dirichlet_rew = np.zeros((trials, T, self.nr, self.nh, self.nc))
        self.observations = np.zeros((trials, T), dtype = int)
        self.rewards = np.zeros((trials, T), dtype = int)
        self.posterior_context = np.ones((trials, T, self.nc))
        self.posterior_context[:,:,:] = self.prior_context[np.newaxis,np.newaxis,:]
        self.likelihood = np.zeros((trials, T, self.npi, self.nc))
        self.prior_policies = np.zeros((trials, self.npi, self.nc))
        self.prior_policies[:] = prior_policies[np.newaxis,:,:]
        self.posterior_actions = np.zeros((trials, T-1, self.na))
        self.posterior_rewards = np.zeros((trials, T, self.nr))
        self.log_probability = 0


    def reset(self, params, fixed):

        self.actions[:] = 0
        self.posterior_states[:] = 0
        self.posterior_policies[:] = 0
        self.posterior_dirichlet_pol[:] = 0
        self.posterior_dirichlet_rew[:] =0
        self.observations[:] = 0
        self.rewards[:] = 0
        self.posterior_context[:,:,:] = self.prior_context[np.newaxis,np.newaxis,:]
        self.likelihood[:] = 0
        self.posterior_actions[:] = 0
        self.posterior_rewards[:] = 0
        self.log_probability = 0

        self.perception.reset(params, fixed)


    def update_beliefs(self, tau, t, observation, reward, response):
        self.observations[tau,t] = observation
        self.rewards[tau,t] = reward

        if t == 0:
            self.possible_polcies = np.arange(0,self.npi,1).astype(np.int32)
        else:
            possible_policies = np.where(self.policies[:,t-1]==response)[0]
            self.possible_polcies = np.intersect1d(self.possible_polcies, possible_policies)
            self.log_probability += ln(self.posterior_actions[tau,t-1,response])

        self.posterior_states[tau, t] = self.perception.update_beliefs_states(
                                         tau, t,
                                         observation,
                                         reward,
                                         self.policies,
                                         self.possible_polcies)

        #update beliefs about policies
        self.posterior_policies[tau, t], self.likelihood[tau,t] = self.perception.update_beliefs_policies(tau, t)

        if tau == 0:
            prior_context = self.prior_context
        else: #elif t == 0:
            prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))
            print(prior_context)
#            else:
#                prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])

        if self.nc>1 and t>0:
            self.posterior_context[tau, t] = \
            self.perception.update_beliefs_context(tau, t, \
                                                   reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   prior_context, \
                                                   self.policies)
        elif self.nc>1 and t==0:
            self.posterior_context[tau, t] = prior_context
        else:
            self.posterior_context[tau,t] = 1

        # print(tau,t)
        # print("prior", prior_context)
        # print("post", self.posterior_context[tau, t])

        if t < self.T-1:
            post_pol = np.dot(self.posterior_policies[tau, t], self.posterior_context[tau, t])
            self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t, post_pol)

        if t == self.T-1 and self.learn_habit:
            self.posterior_dirichlet_pol[tau], self.prior_policies[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
                                                            self.posterior_policies[tau,t], \
                                                            self.posterior_context[tau,t])

        if False:
            self.posterior_rewards[tau, t-1] = np.einsum('rsc,spc,pc,c->r',
                                                  self.perception.generative_model_rewards,
                                                  self.posterior_states[tau,t,:,t],
                                                  self.posterior_policies[tau,t],
                                                  self.posterior_context[tau,t])
        #if reward > 0:
        if self.learn_rew:
            self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
                                                            reward, \
                                                   self.posterior_states[tau, t], \
                                                   self.posterior_policies[tau, t], \
                                                   self.posterior_context[tau,t])

    def generate_response(self, tau, t):

        #get response probability
        posterior_states = self.posterior_states[tau, t]
        posterior_policies = np.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, 0])
        posterior_policies /= posterior_policies.sum()
        avg_likelihood = np.einsum('pc,c->p', self.likelihood[tau,t], self.posterior_context[tau, 0])
        avg_likelihood /= avg_likelihood.sum()
        prior = np.einsum('pc,c->p', self.prior_policies[tau-1], self.posterior_context[tau, 0])
        prior /= prior.sum()
        #print(self.posterior_context[tau, t])
        non_zero = posterior_policies > 0
        controls = self.policies[:, t]#[non_zero]
        actions = np.unique(controls)
        # posterior_policies = posterior_policies[non_zero]
        # avg_likelihood = avg_likelihood[non_zero]
        # prior = prior[non_zero]

        self.actions[tau, t] = self.action_selection.select_desired_action(tau,
                                        t, posterior_policies, controls, avg_likelihood, prior)


        return self.actions[tau, t]


    def estimate_action_probability(self, tau, t, posterior_policies):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[self.policies[:,t] == a].sum()


        return control_prob





# arr_type = "torch"
# if arr_type == "numpy":
#     import numpy as ar
#     array = ar.array
# else:
#     import torch as ar
#     array = ar.tensor
# import numpy as np
# from perception import HierarchicalPerception
# from misc import ln, softmax, own_logical_and
# import scipy.special as scs


# try:
#     from inference_twostage import device
# except:
#     device = ar.device("cpu")

# class FittingAgent(object):
#     def __init__(self, perception, action_selection, policies,
#                  prior_states = None, prior_policies = None,
#                  prior_context = None,
#                  learn_habit = False,
#                  learn_rew = False,
#                  trials = 1, T = 4, number_of_states = 6,
#                  number_of_rewards = 3,
#                  number_of_policies = 8, number_of_planets = None):

#         #set the modules of the agent
#         self.perception = perception
#         self.action_selection = action_selection

#         #set parameters of the agent
#         self.npl = number_of_planets
#         self.nh = number_of_states #number of states
#         self.npi = number_of_policies #number of policies
#         self.nr = number_of_rewards
        
#         if prior_context is not None:
#             self.nc = prior_context.size
#         self.T = T
#         self.trials = trials

#         if policies is not None:
#             self.policies = policies
#         else:
#             #make action sequences for each policy
#             self.policies = ar.eye(self.npi, dtype = int).to(device)

#         self.possible_polcies = self.policies.clone().detach()

#         self.actions = ar.unique(self.policies).to(device)
#         self.na = len(self.actions)

#         if prior_states is not None:
#             self.prior_states = prior_states
#         else:
#             self.prior_states = ar.ones(self.nh).to(device)
#             self.prior_states /= self.prior_states.sum()

#         if prior_context is not None:
#             self.prior_context = prior_context
#             self.nc = prior_context.shape[0]
#         else:
#             self.prior_context = ar.ones(1).to(device)
#             self.nc = 1
        
#         if prior_policies is not None:
#             self.prior_policies = prior_policies
#         else:
#             self.prior_policies = ar.ones((self.npi)).to(device)/self.npi
#         self.learn_habit = learn_habit
#         self.learn_rew = learn_rew

#         #set various data structures
#         self.actions = ar.zeros((trials, T), dtype = int).to(device)
#         self.observations = ar.zeros((trials, T), dtype = int).to(device)
#         self.rewards = ar.zeros((trials, T), dtype = int).to(device)
#         self.posterior_actions = ar.zeros((trials, T-1, self.na)).to(device)
#         self.posterior_rewards = ar.zeros((trials, T, self.nr)).to(device)
#         self.log_probability = 0
#         if hasattr(self.perception, 'generative_model_context'):
#             self.context_obs = ar.zeros(trials, dtype=int).to(device)
#         self.posterior_states = ar.zeros((trials, T, self.nh, T, self.npi, self.nc)).to(device)
#         self.posterior_policies = ar.zeros((trials, T, self.npi, self.nc)).to(device)
#         self.posterior_dirichlet_pol = ar.zeros((trials, self.npi, self.nc)).to(device)
#         if not number_of_planets is None:
#             self.posterior_dirichlet_rew = ar.zeros((trials, T, self.nr, self.arl, self.nc)).to(device)
#         else:
#             self.posterior_dirichlet_rew = ar.zeros((trials, T, self.nr, self.nh, self.nc)).to(device)
#         self.prior_policies = ar.tile(self.prior_policies[None,:,:], (trials, 1,1))

#         self.posterior_context = ar.ones((trials, T, self.nc)).to(device)
#         self.posterior_context[:,:,:] = ar.tile(self.prior_context[None,None,:], (trials, T, 1)) 
#         self.likelihood = ar.zeros((trials, T, self.npi, self.nc)).to(device)
#         self.prior_actions = ar.zeros([trials, T-1, ar.unique(self.policies).size]).to(device)
#         self.outcome_suprise = ar.zeros([trials, T, self.nc]).to(device)
#         self.policy_entropy = ar.zeros([trials, T, self.nc]).to(device)
#         self.context_obs_surprise = ar.zeros([trials, T, self.nc]).to(device)
        
#     def reset(self, param_dict):

#         self.actions = ar.zeros((self.trials, self.T), dtype = int).to(device)
#         self.observations = ar.zeros((self.trials, self.T), dtype = int).to(device)
#         self.rewards = ar.zeros((self.trials, self.T), dtype = int).to(device)
#         self.posterior_actions = ar.zeros((self.trials, self.T-1, self.na)).to(device)
#         self.posterior_rewards = ar.zeros((self.trials, self.T, self.nr)).to(device)
#         self.posterior_states  = ar.zeros((self.trials, self.T, self.nh, self.T, self.npi, self.nc)).to(device)
#         self.posterior_policies = ar.zeros((self.trials, self.T, self.npi, self.nc)).to(device)
#         self.posterior_dirichlet_pol  = ar.zeros((self.trials, self.npi, self.nc)).to(device)
#         self.posterior_dirichlet_rew[:] = ar.zeros((self.trials, self.T, self.nr, self.nh, self.nc)).to(device)
#         self.posterior_context = ar.ones((self.trials, self.T, self.nc)).to(device)
#         self.likelihood[:] = 0
#         self.log_probability = 0
#         if hasattr(self.perception, 'generative_model_context'):
#             self.context_obs = ar.zeros(self.trials, dtype=int).to(device)
#         self.set_parameters(**param_dict)
#         self.perception.reset()

#     # code set up to look at unique planet types
#     #this function creates a reward matrix for the given planet constelation
#     def initiate_planet_rewards(self):
        
#         gen_mod_rewards = ar.zeros([self.nr, self.nh, self.nc])
#         for p in range(self.nh):
#             gen_mod_rewards[:,p,:] =\
#             self.perception.generative_model_rewards[:,self.planets[p],:]
        
#         return gen_mod_rewards

#     def update_beliefs(self, tau, t, observation, reward, response, context=None):
        
#         self.observations[tau,t] = observation
#         self.rewards[tau,t] = reward
#         self.perception.planets = self.planets
#         if context is not None:
#             self.context_obs[tau] = context

#         if t == 0:
#             self.possible_polcies = ar.arange(0,self.npi,1).astype(np.int32)
#         else:
#             possible_policies = np.where(self.policies[:,t-1]==response)[0]
#             self.possible_polcies = np.intersect1d(self.possible_polcies, possible_policies)
#             self.log_probability += ln(self.posterior_actions[tau,t-1,response])

#         self.perception.current_gen_model_rewards = self.initiate_planet_rewards()
#         self.posterior_states[tau, t] = self.perception.update_beliefs_states(
#                                          tau, t,
#                                          observation,
#                                          reward,
#                                          self.policies,
#                                          self.possible_polcies)

#         #!#print("start", self.world.environment.starting_position[tau],"\n")
#         #!#print("posterior over states for two contexts")
#         #!#print(self.posterior_states[tau,t,:,t,:,0].T)
#         #!#print("\n")
#         #!#print(self.posterior_states[tau,t,:,t,:,1].T)
#         #update beliefs about policies
#         self.posterior_policies[tau, t], self.likelihood[tau,t] = self.perception.update_beliefs_policies(tau, t)
        

#         #!#print("\nself.posterior_policies[tau, t]\n", self.posterior_policies[tau, t])
#         #!#print("planet conf", self.world.environment.planet_conf[tau])
#         #!#print("c1: ", self.policies[np.argmax(self.posterior_policies[tau, t][:,0])])
#         #!#print("c2: ", self.policies[np.argmax(self.posterior_policies[tau, t][:,1])])

#         if tau == 0:
#             prior_context = self.prior_context
#         else: #elif t == 0:
#             prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau-1, -1]).reshape((self.nc))

# #            else:
# #                prior_context = np.dot(self.perception.transition_matrix_context, self.posterior_context[tau, t-1])


#         if self.nc>1 and t>=0:
            
#             if hasattr(self, 'context_obs'): 
#                 c_obs = self.context_obs[tau]
#             else:
#                 c_obs = None

#             self.posterior_context[tau, t,:], self.outcome_suprise[tau, t,:],\
#             self.policy_entropy[tau,t,:], self.context_obs_surprise[tau,t,:] = self.perception.update_beliefs_context(tau, t, \
#                                                 reward, \
#                                                 self.posterior_states[tau, t], \
#                                                 self.posterior_policies[tau, t], \
#                                                 prior_context, \
#                                                 self.policies,\
#                                                 context=c_obs)
#         else:
#             self.posterior_context[tau,t] = 1
        
#         if t < self.T-1:
#             post_pol = np.dot(self.posterior_policies[tau, t], self.posterior_context[tau, t])
#             self.posterior_actions[tau, t] = self.estimate_action_probability(tau, t, post_pol)

#         if t == self.T-1 and self.learn_habit:
#             self.posterior_dirichlet_pol[tau], self.prior_policies[tau] = self.perception.update_beliefs_dirichlet_pol_params(tau, t, \
#                                                             self.posterior_policies[tau,t], \
#                                                             self.posterior_context[tau,t])

#         # if False:
#         #     self.posterior_rewards[tau, t-1] = np.einsum('rsc,spc,pc,c->r',
#         #                                           self.perception.generative_model_rewards,
#         #                                           self.posterior_states[tau,t,:,t],
#         #                                           self.posterior_policies[tau,t],
#         #                                           self.posterior_context[tau,t])
#         #if reward > 0:
#         if self.learn_rew and t>0:#==self.T-1:
#             if hasattr(self,  'trial_type'):
#                 if not self.trial_type[tau] == 2:
#                     self.posterior_dirichlet_rew[tau,t] = self.perception.update_beliefs_dirichlet_rew_params(tau, t, \
#                                                             reward, \
#                                                             self.posterior_states[tau, t], \
#                                                             self.posterior_policies[tau, t], \
#                                                             self.posterior_context[tau,t])
#             else:
#                 raise('Agent not extinguishing reward!')

#     def generate_response(self, tau, t):

#         #get response probability
#         posterior_states = self.posterior_states[tau, t]
#         posterior_policies = np.einsum('pc,c->p', self.posterior_policies[tau, t], self.posterior_context[tau, t])
#         posterior_policies /= posterior_policies.sum()
        
#         # average likelihood refers to averaging over contexts
#         avg_likelihood = np.einsum('pc,c->p', self.likelihood[tau,t], self.posterior_context[tau, t])
#         avg_likelihood /= avg_likelihood.sum()
#         prior = np.einsum('pc,c->p', self.prior_policies[tau-1], self.posterior_context[tau, t])
#         prior /= prior.sum()
#         self.prior_actions = self.estimate_action_probability(tau,t,prior)
#         #print(self.posterior_context[tau, t])
#         non_zero = posterior_policies > 0
#         controls = self.policies[:, t]#[non_zero]
#         actions = np.unique(controls)
#         # posterior_policies = posterior_policies[non_zero]
#         # avg_likelihood = avg_likelihood[non_zero]
#         # prior = prior[non_zero]

#         self.actions[tau, t] = self.action_selection.select_desired_action(tau,
#                                         t, posterior_policies, controls, avg_likelihood, prior)


#         return self.actions[tau, t]


#     def estimate_action_probability(self, tau, t, posterior_policies):

#         #estimate action probability
#         control_prob = np.zeros(self.na)
#         for a in range(self.na):
#             control_prob[a] = posterior_policies[self.policies[:,t] == a].sum()


#         return control_prob




# ..................................''.......                 ................................   .....
# .................................''.................     ............'''.........,:;'..       ......
# ....................................';;:;,,'''.''''.........';cc:,...','.........;c:,.        ......
# .............................''..':oxkkkxdooollclccc::;,,,,;:lodddoc;'...........;c:,.              
# ............................;;',lkKKKKKKK00OOkkxxxdddoolllcllodddddxdl;'...  ....:c:,.              
# ...........................,;,:kKXXXXXXKKK000OOkkxxxxdddooooooodddddodoc,'..   ..:c:'.              
# ..........................,;;lOXXXXXXXKKK00OOOkkxxddddddooooooooooolllllc,'..  .':c;'.              
# .........................,;;oOKXXXKKKK000OOkkkxxddddddooooooollllllcccccc:'..  .':c;..              
# .........................;:cx0KKKKKK000Okkxxxddddddoooloooolllccc:::ccc::;,..  .'::;..       ..     
# ........................,:cokO0KK0000OOkxxddoooooooollllllllccc::;;;:::;;;'..   .;:,...      ..     
# ........................,:coxOOOOOOOOOkxddooooolllllllllooollc::;;;;;;;,,,'.    .;;'..  ..          
# ........................':clxkkkOOOOOOkxxxdddoooooooooooddoolc::;;;,,,,,,,..    .;:'.               
#                  ........,:cdkkkkkOOOkkxxxxxddooooooodddddollc:;;;,,''''''..    .::'.               
#                       ....':dkkkkkkOkkxxxxdddoooooddddddoolc::;;;;,,'''''...    .;:'.               
#                          .'cdkOOOOOOkxxxdoolllllloooooolcc::;;,,,;;,,;;,'...    .;:'.               
#                          .'lxkO00Okddooollc::::cccccccc:;,,,,'''''',;;;;,...    .;;..               
#                       .. .'lkkdc:;'............',,,,,,,...            ......    .;;..               
#                        ....co:....             ........                  ...    .:;..               
#   ... ...           .;;'..'lo;...               .:cc;'.                  ...   .';,....             
#        ..        ...cdddo:,lkl...  .           .cddoo:.                  ...  .......               
#        ..       .',:c:coo;'lkd;......         .:ddoooc,.                ..'......''..               
#        .        .,:clcoxo;,okkd:,'......  ...,coxxxdl:,'.             ...''...'''''.                
#        .        ..;loodol:;okkkkxdolc:;,'',;:odxxxxxoc;,'..............'''....',,,..                
#       .......    .';cddll:,lxxxxxxdolc:::cloddxkkkxdoc;,'....................',,,,..                
#  ..........'..  .....cxxdl,:dxddooolllloddddxxxxxdooc;,,''''''''.............,;,,,..      .         
#  ........';;,........'oOkdlcodooollloodxxddool;;;,'''....'''',''''''........',,'''..   ....         
#    ...................'oxollddoolllloodxddolc;..''.........,,,,,,'''........'',,,'.. ......     .. .
#       .    .........   .:looddoollloddxxxdodddlc:,........,,,,,,,''........'''',,'..   ...      ....
#       .    ..',,'..     .;doodoolloodxkkxxxxxdddl,..','''',,,,,,,''............,,...                
#    ...''...........      'lxdddollodxkkxxxdddoooc;',:;,''''''''''''........  ..,,..                 
#   ..,:::,..........      .:xxddollodxxxxdollc:;,''.''........'''''........   ..,,..                 
#  ..,;::;,... ....         ,dxdoollodxxdl:;;;;;;;;,,,''...................   ..',,..                 
#  ...''........            .okdllllodddl::llooooc:;;;,,''................     .';,...                
#    .......                .ckxdolcloolcclolc:;'..........................    .,;,.................. 
#    ...                    .;dxxdolllllcc::;''............................   ..,;,...................
#   ....              ..  ...'ldxdoolllll:;;,'',;;;,,,'.........................;;;,'''''''''''''''''.
# ............     ..........'lxxdoolcccc:::::clc:;,,,,'''...............''....';;;,,,,,,,,,,,,,,,,,''
# .........'..........''''''',dOkxdolc:;;;:::cc:;,,'''''.................,,,,,,,;;;;,,,,,;;;;;;;;;;,,,
# ,....',;;;;;,'...'',;;;;;;;:okOkdolcc:,,;;::;;,''''.....................,;;;;;::::;;;;;:::ccc:::::;;
# :;;;;:::ccccc:::::ccccccc::;;d0kdollc:;;;:::;,''.....................';..,;:;:::::::::::::::::::::::
# ccccccccccccc:cc:cc::::c:;;,,xN0xdollc:;;;;cc:;,''''''...............':'..',;;;;;;;;;;::::::::;;;;::
# ;;;;;;::::;;,,,,,,,,,,;;;,,''dNN0xoolc:;;;;:::;,,'',;,'..............,l,....';::::::::::::::::::::::
# ;;;::::::::;;;;;;;;;;;;;,,,'.cKNXKkdoc::;;;;:;;,,,,,;,'.............,lo;. ....,;;::;;;;;;;;;;;;;;;;;
# :::::::::::::::c::::;,,,,,''.,xXXKKOxl:::;;;;;;;,,',,,'...........';lll:.  .....',,,;;;:::::::::::::
# ccc::::::::::::;,,,'''',,,''..;kKK00Okdlc:;;;,,;,,'''''..........':loll:.  ...........',,,,;;;::::::
# ccccccccc::;;;,'..'..''',,''...:k000OOOkxoc:,,,;,,,''''.........,coolllc.  ..................''',,;;
# ccccc::;;,,,,'.......'''''''....:xOOkkkkxxxdlc:;,''''''....'..,cooooollc'. .........................
# cc::;;,,'''''........''''''......:oxkkkkxxxxxddol::;,'..''.';lddddooollc,.  ........................
# :;;,,''''.............'''''......';cdxkkxxxxxxddddddolc;;;;cxkxddddollcc;.. ........................
# ,,''''................'','''......,;coxkkkxxxxxxddxxkkko;,..:xxxddoollll:'.  .......................
