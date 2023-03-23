"""This module contains the World class that defines interactions between
the environment and the agent. It also keeps track of all observations and
actions generated during a single experiment. To initiate it one needs to
provide the environment class and the agent class that will be used for the
experiment.
"""
import numpy as np
from misc import ln
import torch as ar
# from misc_sia import *
ar.set_default_dtype(ar.float64)
ar.set_printoptions(precision=8, threshold=100)

class FittingWorld(object):

    def __init__(self, environment, agent, trials = 1, T = 10):
        #set inital elements of the world to None
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial

        self.free_parameters = {}

        #container for observations
        self.observations = ar.zeros((self.trials, self.T), dtype=ar.int32)

        #container for agents actions
        self.actions = ar.zeros((self.trials, self.T), dtype=ar.int32)

        #container for rewards
        self.rewards = ar.zeros((self.trials, self.T), dtype=ar.int32)
        self.environment.possible_rewards = self.agent.perception.possible_rewards

    def simulate_experiment(self, curr_trials=None, print_thoughts=False):
        """This methods evolves all the states of the world by iterating
        through all the trials and time steps of each trial.
        """
        if curr_trials is not None:
            trials = curr_trials
        else:
            trials = range(self.trials)
        for tau in trials:
            if tau%3 == 0:
                print('breaktpoint')
            for t in range(self.T):
                # if print_thoughts:
                    # print("tau", tau, ", t", t)
                self.__update_world(tau, t)
            # print('tau: ', tau, ' t:', t)
                # print(self.actions[tau])

        # self.convert_to_numpy()

    # def convert_to_numpy(self):
    #     self.agent.perception = convert(self.agent.perception)
    #     self.agent.action_selection = convert(self.agent.action_selection)
    #     self.agent = convert(self.agent)
    #     self.environment = convert(self.environment)
    #     self.rewards = self.rewards.numpy()
    #     self.actions = self.actions.numpy()
    #     self.observations = self.observations.numpy()
    #     self.dec_temp = self.dec_temp.numpy()

    #this is a private method do not call it outside of the class
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the
        whole world. Here we update the hidden state(s) of the environment,
        the perceptual and planning states of the agent, and in parallel we
        generate observations and actions.
        """

        if t==0:
            self.environment.set_initial_states(tau)
            response = None
            if hasattr(self.environment, 'Chi') or \
               self.agent.perception.generative_model_context is not None:

                context = self.environment.generate_context_obs(tau)
            else:
                context = None
        else:
            response = self.actions[tau, t-1]
            self.environment.update_hidden_states(tau, t, response)
            context = None


        self.observations[tau, t] = \
            self.environment.generate_observations(tau, t)

        if t>0:
            self.rewards[tau, t] = self.environment.generate_rewards(tau, t)

        if hasattr(self, 'trial_type'):
            if self.trial_type[tau] == 2:
                reward = 0
        observation = self.observations[tau, t]

        reward = self.rewards[tau, t]

        if self.environment.planet_confs is not None:
            self.agent.perception.planets = self.environment.planet_conf[tau,:]
        # print('planets: ', self.agent.planets)
        # print('start: ', self.environment.starting_position[tau])
        self.agent.update_beliefs(tau, t, observation, reward, response, context)


        if t < self.T-1:
            self.actions[tau, t] = self.agent.generate_response(tau, t)
        else:
            self.actions[tau, t] = -1
        

        if False:
            print( '\n\n','tau, t: ', tau,t)
            print('observation: ', self.environment.planet_conf[tau][self.rewards[tau,t]].numpy())
            print('reward:', self.rewards[tau,t].numpy())
            print('action:', self.actions[tau,t].numpy())

            print('\nposterior policies:')
            print(self.agent.perception.posterior_policies[-1][...,0].numpy().round(5))
            print('\nposterior contexts: ')
            print(self.agent.perception.posterior_contexts[-1][...,0].numpy().round(5))
            print('\nposterior rewards: ')
            for i in range(4):
                print('context ', i)
                print(self.agent.perception.dirichlet_rew_params[-1][...,i,0].numpy())
            print('\ndirichlet_pol_params')
            print(self.agent.perception.dirichlet_pol_params[-1][...,0].numpy())

        if tau == 200:
            print('tau == 200')

class World(object):

    def __init__(self, environment, agent, trials = 1, T = 10):
        #set inital elements of the world to None
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial

        self.free_parameters = {}

        #container for observations
        self.observations = np.zeros((self.trials, self.T), dtype = int)

        #container for agents actions
        self.actions = np.zeros((self.trials, self.T), dtype = int)

        #container for rewards
        self.rewards = np.zeros((self.trials, self.T), dtype = int)
        self.environment.possible_rewards = self.agent.perception.possible_rewards

    def simulate_experiment(self, curr_trials=None, print_thoughts=False):
        """This methods evolves all the states of the world by iterating
        through all the trials and time steps of each trial.
        """
        if curr_trials is not None:
            trials = curr_trials
        else:
            trials = range(self.trials)
        for tau in trials:
            for t in range(self.T):
                if print_thoughts:
                    print("tau", tau, ", t", t)
                self.__update_world(tau, t, print_thoughts=print_thoughts)
                # print('tau: ', tau, ' t:', t)
                # print(self.agent.posterior_dirichlet_rew[tau,t])


    #this is a private method do not call it outside of the class
    def __update_world(self, tau, t, print_thoughts = False):
        """This private method performs a signel time step update of the
        whole world. Here we update the hidden state(s) of the environment,
        the perceptual and planning states of the agent, and in parallel we
        generate observations and actions.
        """

        if t==0:
            self.environment.set_initial_states(tau)
            response = None
            if hasattr(self.environment, 'Chi') or \
               self.agent.perception.generative_model_context is not None:

                context = self.environment.generate_context_obs(tau)
            else:
                context = None
        else:
            response = self.actions[tau, t-1]
            self.environment.update_hidden_states(tau, t, response)
            context = None


        self.observations[tau, t] = \
            self.environment.generate_observations(tau, t)

        if t>0:
            self.rewards[tau, t] = self.environment.generate_rewards(tau, t)

        if hasattr(self, 'trial_type'):
            if self.trial_type[tau] == 2:
                reward = 0

        observation = self.observations[tau, t]

        reward = self.rewards[tau, t]


        if self.environment.planet_conf is not None:
            self.agent.perception.planets = self.environment.planet_conf[tau,:]        # print('planets: ', self.agent.planets)
        # print('start: ', self.environment.starting_position[tau])
        self.agent.update_beliefs(tau, t, observation, reward, response, context)


        if t < self.T-1:
            self.actions[tau, t] = self.agent.generate_response(tau, t)
        else:
            self.actions[tau, t] = -1
        

        # print('\n\n----------')
        # print('tau,t:',tau,t)
        # print('reward, action', self.rewards[tau,t], self.actions[tau,t])
        # print('\nfwd_norms:')
        # for i in range(5):
        #     print('\n', self.agent.perception.fwd_norms[tau,t,i,...])
        # print('\nposterior_states')
        # print(self.agent.posterior_states[tau,t,:,:,0,0])
        # print('\nposterior_policies')
        # print(self.agent.posterior_policies[tau,t])
        # print('\nprior_context')
        # print((self.agent.prior_context_log[tau,t]))
        # print('\nposterior_context')
        # print(self.agent.posterior_context[tau,t])
        # print('\noutcome_suprise')
        # print(self.agent.outcome_suprise[tau,t])
        # print('\npolicy_entropy')
        # print(self.agent.policy_entropy[tau,t])
        # print('\npolicy_surprise')
        # print(self.agent.policy_surprise[tau,t])
        # print('\ncontext_obs_suprise')
        # print(self.agent.context_obs_suprise[tau,t])
        # print('\nposterior_rewards')
        # print(self.agent.posterior_dirichlet_rew[tau,t])
        # print('\ngenerative_model_rewards')
        # print(self.agent.perception.generative_model_rewards[tau,t])
        # print('\ncurr_gen_mod_rewards')
        # print(self.agent.perception.current_gen_model_rewards)
        # print('\prior_policy')
        # print(self.agent.perception.prior_policies[tau])
        if print_thoughts:
            print("response", response)
            if t>0:
                print("rewards", self.rewards[tau, t])
            # print("posterior policies: ", self.agent.posterior_policies[tau,t])
            # print("posterior context: ", self.agent.posterior_context[tau,t])
            if t<self.T-1:
                print("prior actions: ", self.agent.prior_actions)
                print("posterior actions: ", self.agent.posterior_actions[tau,t])
                print("\n")

class World_old(object):

    def __init__(self, environment, agent, trials = 1, T = 10):
        #set inital elements of the world to None
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial

        self.free_parameters = {}

        #container for observations
        self.observations = np.zeros((self.trials, self.T), dtype = int)

        #container for agents actions
        self.actions = np.zeros((self.trials, self.T), dtype = int)

        #container for rewards
        self.rewards = np.zeros((self.trials, self.T), dtype = int)

    def simulate_experiment(self, curr_trials=None):
        """This methods evolves all the states of the world by iterating
        through all the trials and time steps of each trial.
        """
        if curr_trials is not None:
            trials = curr_trials
        else:
            trials = range(self.trials)
        for tau in trials:
            for t in range(self.T):
                self.__update_world(tau, t)


    def estimate_par_evidence(self, params, method='MLE'):


        val = np.zeros(params.shape[0])
        for i, par in enumerate(params):
            if method == 'MLE':
                val[i] = self.__get_log_likelihood(par)
            else:
                val[i] = self.__get_log_jointprobability(par)

        return val

    def fit_model(self, bounds, n_pars, method='MLE'):
        """This method uses the existing observation and response data to
        determine the set of parameter values that are most likely to cause
        the meassured behavior.
        """

        inference = Inference(ftol = 1e-4, xtol = 1e-8, bounds = bounds,
                           opts = {'np': n_pars})

        if method == 'MLE':
            return inference.infer_posterior(self.__get_log_likelihood)
        else:
            return inference.infer_posterior(self.__get_log_jointprobability)


    #this is a private method do not call it outside of the class
    def __get_log_likelihood(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()

        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)

        return ln(self.agent.asl.control_probability[p1, p2, p3]).sum()

    def __get_log_jointprobability(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()

        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)

        ll = ln(self.agent.asl.control_probability[p1, p2, p3]).sum()

        return  ll + self.agent.log_prior()

    #this is a private method do not call it outside of the class
    def __update_model(self):
        """This private method updates the internal states of the behavioral
        model given the avalible set of observations and actions.
        """

        for tau in range(self.trials):
            for t in range(self.T):
                if t == 0:
                    response = None
                else:
                    response = self.actions[tau, t-1]

                observation = self.observations[tau,t]

                self.agent.update_beliefs(tau, t, observation, response)
                self.agent.plan_behavior(tau, t)
                self.agent.estimate_response_probability(tau, t)

    #this is a private method do not call it outside of the class
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the
        whole world. Here we update the hidden state(s) of the environment,
        the perceptual and planning states of the agent, and in parallel we
        generate observations and actions.
        """

        if t==0:
            self.environment.set_initial_states(tau)
            response = None
        else:
            response = self.actions[tau, t-1]
            self.environment.update_hidden_states(tau, t, response)

        self.observations[tau, t] = \
            self.environment.generate_observations(tau, t)

        if t>0:
            self.rewards[tau, t] = self.environment.generate_rewards(tau, t)

        observation = self.observations[tau, t]

        reward = self.rewards[tau, t]
        self.agent.update_beliefs(tau, t, observation, reward, response)


        if t < self.T-1:
            self.actions[tau, t] = self.agent.generate_response(tau, t)
        else:
            self.actions[tau, t] = -1

class FakeWorld(object):

    def __init__(self, agent, observations, rewards, actions, trials = 1, T = 10, log_prior=0):
        #set inital elements of the world to None
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial

        self.free_parameters = {}

        #container for observations
        self.observations = observations

        #container for agents actions
        self.actions = actions

        #container for rewards
        self.rewards = rewards

        self.log_prior = log_prior

        self.like_actions = ar.zeros((trials,T-1))
        self.like_rewards = ar.zeros((trials,T-1))
        self.log_policies = []
        self.log_context = []
        self.log_prior_pols = []
        self.log_post_pols = []

    def simulate_experiment(self, curr_trials=None):
        """This methods evolves all the states of the world by iterating
        through all the trials and time steps of each trial.
        """
        if curr_trials is not None:
            trials = curr_trials
        else:
            trials = range(self.trials)
        for tau in trials:
            # print('\n')
            for t in range(self.T):
                # print(tau, t)
                self.__update_model(tau, t)

    def __simulate_agent(self):
        """This methods evolves all the states of the world by iterating
        through all the trials and time steps of each trial.
        """

        for tau in range(self.trials):

            for t in range(self.T):
                self.__update_model(tau, t)


    def estimate_par_evidence(self, params, fixed):


        val = self.__get_log_jointprobability(params, fixed)

        return val

    def fit_model(self, params, fixed, test_trials):
        """This method uses the existing observation and response data to
        determine the set of parameter values that are most likely to cause
        the meassured behavior.
        """
        self.like_actions = ar.zeros((self.trials,self.T-1))
        self.like_rewards = ar.zeros((self.trials,self.T-1))
        self.agent.reset(params, fixed)

        self.__simulate_agent()

        #print(self.like_actions)
        #print(self.like_rewards)

        like = self.like_actions[test_trials].prod() * self.like_rewards[test_trials].prod()

        return like #self.agent.posterior_actions.squeeze(), self.agent.posterior_rewards[:,1]

    def __get_log_jointprobability(self, params, fixed):

        self.agent.reset(params, fixed)

        self.__simulate_agent()

        p1 = ar.tile(ar.arange(self.trials), (self.T-1, 1)).T
        p2 = ar.tile(ar.arange(self.T-1), (self.trials, 1))
        p3 = self.actions.astype(int)
        #self.agent.log_probability
        ll = self.agent.log_probability#ln(self.agent.posterior_actions[p1, p2, p3].prod())

        return  ll + self.log_prior

    #this is a private method do not call it outside of the class
    def __update_model(self, tau, t):
        """This private method updates the internal states of the behavioral
        model given the avalible set of observations and actions.
        """
        if (tau == 0 and t == 0):
            print(self.actions[:10,:])
        context = self.context_obs[tau]
        self.agent.planets = self.planets[tau]
        if t==0:
            response = None
        else:
            response = self.actions[tau, t-1]

            self.like_actions[tau,t-1] = self.agent.posterior_actions[tau, t-1, response]

        observation = self.observations[tau, t]

        reward = self.rewards[tau, t]

        self.agent.update_beliefs(tau, t, observation, reward, response, context)
        if (t == self.T-1):
            print('\n',tau,t)

            try:
                n_digits = 3
                # ppls = self.agent.perception.posterior_policies[-4][:,:,0]
                # rounded = (ppls * 10**n_digits).round() / (10**n_digits)
                ppls = self.agent.perception.posterior_contexts[-1][:,0]
                print(ppls)
                ppls = (self.agent.perception.posterior_actions[-1][:,0])
                print(ppls)
                
                # self.log_context.append(int(ar.argmax( self.agent.perception.posterior_contexts[-1][:,0])))
                # self.log_policies.append(ar.argmax(self.agent.perception.posterior_policies[-1][:,:,0],axis=0).tolist())
                # self.log_post_pols.append(self.agent.perception.posterior_policies[-4][:,:,0].tolist())
                # self.log_prior_pols.append(self.agent.perception.prior_policies[-1][:,:,0].tolist())

                # print('context: ', self.log_context[-1])
                # print('policy: ', self.log_policies[-1])

            except:
                # print(self.agent.posterior_policies[tau,t].round(4))


                # self.log_context.append(int(np.argmax(self.agent.posterior_context[tau,t,:])))
                # self.log_policies.append(np.argmax(self.agent.posterior_policies[tau,t,:],axis=0).tolist())
                # self.log_post_pols.append(self.agent.posterior_policies[tau,0,:].tolist())
                # self.log_prior_pols.append(self.agent.prior_policies[:,:,0].tolist())
                # print(self.agent.posterior_actions.shape)
                print(self.agent.posterior_context[tau,t,:])
                print(self.agent.posterior_actions[tau,t-1,:])
                # print('context: ', self.log_context[-1])
                # print('policy: ', self.log_policies[-1])


        # if t==1:
        #     self.like_rewards[tau,t-1] = self.agent.posterior_rewards[tau, t-1, reward]