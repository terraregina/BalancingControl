"""This module contains various experimental environments used for testing
human behavior."""
import numpy as np



class PlanetWorld(object):

    def __init__(self, O, S, R, planet_configurations, starts, contexts, mb = 10, T = 4, ns=6, npl=2, nr=2, na=2):

        self.A = O.copy()                                       # prob dist for generating observations
        self.B = S.copy()                                       # prob dist for state transitions
        self.R = R.copy()                                       # prob dist for reward generation
        self.trials = mb                                        # mb corresponds to number of miniblocks
        self.T = T                                              # miniblock length of 3 actions + initial state = 4
        self.ns = ns                                            # number of unique locations
        self.nr = nr                                            # number of rewards
        self.npl = npl                                          # number of unique planet types
        self.na = na                                            # number of actions
        self.context_cues = np.array(contexts,dtype="int32")    # background colors, look at run_agent_simulation, load vars for coding
        self.planet_conf = planet_configurations                # planet identities for each trial
        self.starting_position = starts                         # initial rocket position for each trial
        # hidden states tracks location and not planet identity
        self.hidden_states = np.zeros([mb, T],
         dtype="int32") 


    def set_initial_states(self, tau):
        self.hidden_states[tau, 0] = self.starting_position[tau]

    def generate_context_obs(self,tau):
        return self.context_cues[tau]

    def generate_observations(self,tau,t):
        
        return self.hidden_states[tau,t]
        

    # this function is specifically tailored to deterministic location transitions
    def update_hidden_states(self,tau, t, action):

        curr_loc =  self.hidden_states[tau,t-1]
        self.hidden_states[tau,t] = np.argmax(self.B[:,curr_loc,action,0])
        #!#print( "curr_loc ", curr_loc, ", action ",action,", target ", self.hidden_states[tau,t])

    def generate_rewards(self,tau,t):
        
        curr_loc = self.hidden_states[tau,t]           # t+1 because we are still at t but have already moved the rocket
        rp = self.R[tau,:,curr_loc]                    # reward probability at current planet
        reward = np.random.choice(self.possible_rewards, p=rp)
        return reward

class GridWorld(object):

    def __init__(self, Omega, Theta, Rho,
                 trials = 1, T = 10, initial_state = 2):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating rewards
        self.Rho = Rho.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)

        #set intial state
        self.initial_state = initial_state

    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = self.initial_state

        if tau%100==0:
            print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response)])

    def generate_rewards(self, tau, t):
        
        #generate one sample from multinomial distribution
        curr_loc = self.hidden_states[tau,t]           # t+1 because we are still at t but have already moved the rocket
        rp = self.R[tau,0,curr_loc]                      # reward probability at current planet
        reward = np.random.binomial(n=1, p=rp)
        
        rewards = [1,-1,0]
        reward = rewards[t]
        return reward

"""
test: please ignore
"""
class FakeGridWorld(object):

    def __init__(self, Omega, Theta,
                 hidden_states, trials = 1, T = 10):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)
        self.hidden_states[:] = np.array([hidden_states for i in range(trials)])

    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 1

        #print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response)])


class MultiArmedBandid(object):

    def __init__(self, Omega, Theta, Rho,
                 trials = 1, T = 10):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating rewards
#        self.Rho = np.zeros((trials, Rho.shape[0], Rho.shape[1]))
#        self.Rho[0] = Rho.copy()
        self.Rho = Rho.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        self.nh = Theta.shape[0]

#        self.changes = np.array([0.01, -0.01])

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)

        self.trials = trials

    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 0

#        if tau%100==0:
#            print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response)])

    def generate_rewards(self, tau, t):
        #generate one sample from multinomial distribution
        r = np.random.choice(self.Rho.shape[1], p = self.Rho[tau, :, self.hidden_states[tau, t]])

#        if tau < self.trials-1:
#            #change Rho slowly
#            change = np.random.choice(self.changes, size=self.nh-1)
#            self.Rho[tau+1,0,1:] = self.Rho[tau,0,1:] + change
#            self.Rho[tau+1,1,1:] = self.Rho[tau,1,1:] - change
#            self.Rho[tau+1][self.Rho[tau+1] > 1.] = 1.
#            self.Rho[tau+1][self.Rho[tau+1] < 0.] = 0.

        return r

class TaskSwitching(object):

    def __init__(self, Omega, Theta, Rho, Chi, start_states, contexts,
                 trials = 1, T = 10, correct_choice=None, congruent=None,
                 num_in_run=None):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating rewards
#        self.Rho = np.zeros((trials, Rho.shape[0], Rho.shape[1]))
#        self.Rho[0] = Rho.copy()
        self.Rho = Rho.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        self.nh = Theta.shape[0]
        
        self.Chi = Chi.copy()

#        self.changes = np.array([0.01, -0.01])

        assert(len(start_states==trials))

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)
        self.hidden_states[:,0] = start_states
        
        self.contexts = contexts.copy().astype(int)

        self.trials = trials
        
        if correct_choice is not None:
            self.correct_choice = correct_choice
        if congruent is not None:
            self.congruent = congruent
        if num_in_run is not None:
            self.num_in_run = num_in_run

    def set_initial_states(self, tau):
        #start in lower corner
        #self.hidden_states[tau, 0] = 0
        pass

#        if tau%100==0:
#            print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]
        current_context = self.contexts[tau]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response), current_context])

    def generate_rewards(self, tau, t):
        #generate one sample from multinomial distribution
        r = np.random.choice(self.Rho.shape[1], p = self.Rho[tau, :, self.hidden_states[tau, t]])

#        if tau < self.trials-1:
#            #change Rho slowly
#            change = np.random.choice(self.changes, size=self.nh-1)
#            self.Rho[tau+1,0,1:] = self.Rho[tau,0,1:] + change
#            self.Rho[tau+1,1,1:] = self.Rho[tau,1,1:] - change
#            self.Rho[tau+1][self.Rho[tau+1] > 1.] = 1.
#            self.Rho[tau+1][self.Rho[tau+1] < 0.] = 0.

        return r
    
    def generate_context_obs(self, tau):
        
        c = np.random.choice(self.Chi.shape[0], p=self.Chi[self.contexts[tau]])
        return c
    

class Flanker(object):

    def __init__(self, Omega, Theta, Rho, Chi, start_states, contexts, flankers,
                 trials = 1, T = 10, correct_choice=None, congruent=None):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating rewards
#        self.Rho = np.zeros((trials, Rho.shape[0], Rho.shape[1]))
#        self.Rho[0] = Rho.copy()
        self.Rho = Rho.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        self.nh = Theta.shape[0]
        
        self.Chi = Chi.copy()

#        self.changes = np.array([0.01, -0.01])

        assert(len(start_states==trials))

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)
        self.hidden_states[:,0] = start_states
        
        self.contexts = contexts.copy().astype(int)
        
        self.flankers = flankers.copy()

        self.trials = trials
        
        if correct_choice is not None:
            self.correct_choice = correct_choice
        if congruent is not None:
            self.congruent = congruent

    def set_initial_states(self, tau):
        #start in lower corner
        #self.hidden_states[tau, 0] = 0
        pass

#        if tau%100==0:
#            print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]
        current_context = self.contexts[tau]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response), current_context])

    def generate_rewards(self, tau, t):
        #generate one sample from multinomial distribution
        r = np.random.choice(self.Rho.shape[1], p = self.Rho[tau, :, self.hidden_states[tau, t]])

#        if tau < self.trials-1:
#            #change Rho slowly
#            change = np.random.choice(self.changes, size=self.nh-1)
#            self.Rho[tau+1,0,1:] = self.Rho[tau,0,1:] + change
#            self.Rho[tau+1,1,1:] = self.Rho[tau,1,1:] - change
#            self.Rho[tau+1][self.Rho[tau+1] > 1.] = 1.
#            self.Rho[tau+1][self.Rho[tau+1] < 0.] = 0.

        return r
    
    def generate_context_obs(self, tau):
        
        c = np.random.choice(self.Chi.shape[0], p=self.Chi[self.contexts[tau]])
        return c


class TMaze(object):

    def __init__(self, Omega, Theta, Rho,
                 trials = 1, T = 10):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating rewards
#        self.Rho = np.zeros((trials, Rho.shape[0], Rho.shape[1]))
#        self.Rho[0] = Rho.copy()
        self.Rho = Rho.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        self.nh = Theta.shape[0]

#        self.changes = np.array([0.01, -0.01])

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)

        self.trials = trials

    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 0

#        if tau%100==0:
#            print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response)])

    def generate_rewards(self, tau, t):
        #generate one sample from multinomial distribution
        r = np.random.choice(self.Rho.shape[1], p = self.Rho[tau, :, self.hidden_states[tau, t]])

        return r


class TwoStep(object):

    def __init__(self, Omega, Theta, Rho,
                 trials = 1, T = 10):

        #set probability distribution used for generating observations
        self.Omega = Omega.copy()

        #set probability distribution used for generating rewards
        self.Rho = np.zeros((trials, Rho.shape[0], Rho.shape[1]))
        self.Rho[0] = Rho.copy()

        #set probability distribution used for generating state transitions
        self.Theta = Theta.copy()

        self.nh = Theta.shape[0]

        self.changes = np.array([0.01, -0.01])

        #set container that keeps track the evolution of the hidden states
        self.hidden_states = np.zeros((trials, T), dtype = int)

        self.trials = trials
        self.T = T

    def set_initial_states(self, tau):
        #start in lower corner
        self.hidden_states[tau, 0] = 0

        if tau%100==0:
            print("trial:", tau)


    def generate_observations(self, tau, t):
        #generate one sample from multinomial distribution
        o = np.random.multinomial(1, self.Omega[:, self.hidden_states[tau, t]]).argmax()
        return o


    def update_hidden_states(self, tau, t, response):

        current_state = self.hidden_states[tau, t-1]

        self.hidden_states[tau, t] = np.random.choice(self.Theta.shape[0],
                          p = self.Theta[:, current_state, int(response)])

    def generate_rewards(self, tau, t):
        #generate one sample from multinomial distribution
        r = np.random.choice(self.Rho.shape[1], p = self.Rho[tau, :, self.hidden_states[tau, t]])

        if (tau < self.trials-1) and t == self.T-1:
            #change Rho slowly
            self.Rho[tau+1] = self.Rho[tau]
            change = np.random.choice(self.changes, size=self.nh - 3)
            self.Rho[tau+1,0,3:] = self.Rho[tau+1,0,3:] + change
            self.Rho[tau+1,1,3:] = self.Rho[tau+1,1,3:] - change
            self.Rho[tau+1,:,3:][self.Rho[tau+1,:,3:] > 1.] = 1.
            self.Rho[tau+1,:,3:][self.Rho[tau+1,:,3:] < 0.] = 0.

        return r