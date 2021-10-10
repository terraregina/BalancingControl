# from misc import ln, logBeta, Beta_function
import numpy as np
from statsmodels.tsa.stattools import acovf as acov
import scipy.special as scs
from scipy.stats import entropy
import matplotlib.pylab as plt
from misc import *



'''
_______   _______   __       __ 
/       \ /       \ /  \     /  |
$$$$$$$  |$$$$$$$  |$$  \   /$$ |
$$ |  $$ |$$ |  $$ |$$$  \ /$$$ |
$$ |  $$ |$$ |  $$ |$$$$  /$$$$ |
$$ |  $$ |$$ |  $$ |$$ $$ $$/$$ |
$$ |__$$ |$$ |__$$ |$$ |$$$/ $$ |
$$    $$/ $$    $$/ $$ | $/  $$ |
$$$$$$$/  $$$$$$$/  $$/      $$/ 
                                 
'''
                                 


# drift diffusion random walker
class DDM_RandomWalker(object):

    def __init__(self, trials, T, wd = 1, s = 0.01, alpha = [-1,1], dt = 0.01, max=10000):

        self.dt = 0.01
        self.wd = wd
        self.s = np.sqrt(s)
        self.al = alpha[0]                     # decision boundries
        self.au = alpha[1]
        self.max = 10000
        self.RT = np.zeros([trials,T-1])
        self.walks = []
        self.over_actions = False
        self.type = 'ddm'

    def estimate_action_probability(self, tau, dist_policies, actions, *args):

        #estimate action probability
        na = np.unique(actions.size)
        control_prob = np.zeros(na)

        for a in range(na):
            control_prob[a] = dist_policies[actions == a].sum()

    #    normalize?

        print(control_prob)
        print( control_prob / control_prob.sum())
        return  control_prob 
        # return control_prob/control_prob.sum() 
        

    # def select_desired_action(self, tau, step, prior, likelihood, posterior,  plot=False, *args):
    def select_desired_action(self, tau, step, posterior, controls, *args):
    
        if(args != ()):
            likelihood = args[0]
            prior = args[1]
        dt = self.dt
        max = self.max           
        #     dt = args[2]
        #     max = args[3]
        # else:
        #     dt = self.dt
        #     max = self.max

        # if len(args) > 4:
        #     w,s = args[3,4]
        # else:                           # functions as a free parameter which can reflect noisiness of environment or individual capabilities?
        #     # print("got here")
        #     w,s = [self.w, self.s]      # drift rate, learning dependent on quality of sensory info determining decision
                                        # diffusion rate, noise?


        if self.over_actions:
            # transform a likelihood and prior over policies to a likelihood and prior over actions
            prior = self.estimate_action_probability(tau, prior, controls)
            likelihood = self.estimate_action_probability(tau, likelihood, controls)


        d0 = prior[0] - prior[1]            # diff
        # self.d = A*(prior[0]/prior[1])    # product

                                            
        Q = likelihood[0] - likelihood[1]   # evidence

        walk = [d0]
        d = d0
        t = 0
        # print(s)

        while(self.al < d and d < self.au):

            if(t < max):
                epsilon = np.random.normal(0,self.s)
                d += self.wd*Q*dt + epsilon
                walk.append(d)
                t += 1

            else:
                print('walk finished')
                break


        if (False):
            plt.plot(range(0,t,1),walk)
            plt.show()

        # d > self.au coresponds to action 0
        action = not d > self.au
        self.RT[tau, step] = t
        self.walks.append(walk)
        
        return action

'''
  ______           _______   _______   __       __ 
 /      \         /       \ /       \ /  \     /  |
/$$$$$$  |        $$$$$$$  |$$$$$$$  |$$  \   /$$ |
$$ |__$$ | ______ $$ |__$$ |$$ |  $$ |$$$  \ /$$$ |
$$    $$ |/      |$$    $$< $$ |  $$ |$$$$  /$$$$ |
$$$$$$$$ |$$$$$$/ $$$$$$$  |$$ |  $$ |$$ $$ $$/$$ |
$$ |  $$ |        $$ |  $$ |$$ |__$$ |$$ |$$$/ $$ |
$$ |  $$ |        $$ |  $$ |$$    $$/ $$ | $/  $$ |
$$/   $$/         $$/   $$/ $$$$$$$/  $$/      $$/ 
'''
                                                   
                                                   
                                                   

''' Implements the Advantage LBA model (van Ravenzwaaij et al, 2019)'''
''' selects a decision from N-alternatives through propagating      '''
''' linear evidence integrators. Decision d_i made when ............'''
''' Basic equation for integrator:                                  '''
''' dx_i = v_ij + s*dW                                              '''
''' v_ij = v0_i + wd*(S_i - S_j) + ws*(S_i + S_j)                   '''
''' dW = N(0,s)'''

'''Parameters:                                                      '''
''' z0_i = starting position                                        '''
''' v0 = urgency                                                    '''
''' wd, ws = advantage and sum term weights                         '''
''' s = standard deviation                                          '''
''' t0 (non-decision time) currently not considered                 '''
''' b = decision threshold                                          '''


# for policy likelihood and post look at perception update_beliefs
# we want the context averaged out, so look into how it is with prior and likelihood
# I can do either the free energy or the likelihood, maybe try both

class AdvantageRacingDiffusionSelector(object):

    def __init__ (self, trials, T, number_of_actions, wd = 1, ws = 0, s = 0.1, b= 1, A = 1, v0 = 1, over_actions=False):
        
        self.s  = np.sqrt(s)      # standard deviation of decision variable
        self.wd = wd     # weight of advantage term
        self.ws = ws     # weight of sum turm 
        self.v0 = v0     # urgency / bias term
        self.b = b       # boundary, needs to be implemented
        self.A = A       # range of possible starting points
        # self.B = b - A   # distance from max starting point to decision boundry

        self.type = 'ardm'
        self.trials = trials
        self.T = T
        self.RT = np.zeros([trials,T-1])
        self.na = number_of_actions
        self.walks = []
        self.episode_walks = []
        self.over_actions = over_actions
        self.prior_as_starting_point = True
        self.sample_other = False
        self.sample_posterior = False


    def reset_beliefs(self):
        pass
        # self.control_probability[:,:,:] = 0


    def set_pars(self, pars):
        pass    


    def log_prior(self):
        return 0

    def estimate_action_probability(self, tau, dist_policies, actions, *args):

        #estimate action probability
        na = np.unique(actions).size
        control_prob = np.zeros(na)

        for a in range(na):
            control_prob[a] = dist_policies[actions == a].sum()

    #    normalize?

        # print(control_prob)
        # print(control_prob /control_prob.sum())
        return  control_prob 

    def select_desired_action(self,tau, t, posterior, controls, *args): #avg_likelihood, prior, dt = 0.01, plot=False):
     # def select_desired_action(self,tau, t, posterior, controls, *args): #avg_likelihood, prior, dt = 0.01, plot=False):

        likelihood = args[0]
        prior = args[1]
        
        if (len(args) == 3):
            dt = args[3]
        else: 
            dt = 0.01


        if self.over_actions:
            prior = self.estimate_action_probability(tau, prior, controls)
            likelihood = self.estimate_action_probability(tau, likelihood, controls)
            posterior = self.estimate_action_probability(tau, posterior, controls)


        npi = prior.size                                               # number of policies
        ni = np.int(np.math.factorial(npi) / np.math.factorial(npi-2)) # P(2,npi), number of integrators


        decision_log = np.zeros([10005,ni])                            # log for propagated decision variable

        if self.prior_as_starting_point:
            decision_log[0,:] = np.repeat(prior, (npi-1))*self.A       # initial conditions for v_ij


        if self.sample_posterior:
            P = posterior
        elif self.sample_other:
            P = likelihood + prior
        else:
            P = likelihood

        Q = np.tile(P, npi)                                            # set up (S_i - S_j) and (S_i + S_j) terms
        Q_diff = np.repeat(P, (npi-1)) - np.delete(Q,[npi*i + i for i in np.arange(npi)])
        Q_sum = np.repeat(P, (npi-1)) + np.delete(Q,[npi*i + i for i in np.arange(npi)])
    
        v0 = np.ones(ni)*self.v0                                       # vectorized urgency term
    
        bound_reached = np.zeros(npi, dtype = bool)                    # decision made when all integrators for a group are above bound b, Winner takes all strategy. 
        integrators_that_crossed_bound = 0                             # if pi = 1,2,3, decision for policy 1 made when S_12, S_13 >= b before other sets
        
                                          
        i = 0

        broken = False
        while (not np.any(integrators_that_crossed_bound == (npi -1))):                 # if still no decision made

            dW = np.random.normal(scale = self.s, size = ni)
            decision_log[i+1,:] = decision_log[i,:] + (v0 + self.wd*Q_diff + self.ws*Q_sum)*dt + dW 
            bound_reached = decision_log[i+1,:].reshape([npi,npi-1]) >= self.b
            integrators_that_crossed_bound = bound_reached.sum(axis=1)
            i+=1

            if (i > 10000):
                broken = True
                print("COULDN'T REACH DECISION IN 10000 STEPS")
                break    

        crossed = np.where(integrators_that_crossed_bound == (npi -1))[0]             # non-zero returns tuple by default

        if (crossed.size > 1):
            # print('Two integrators crossed decision boundary at the same time')
            total_dist_travelled_by_all_integrators = decision_log[i,:].reshape([npi,npi-1]).sum(axis=1)
            dist_by_winners = total_dist_travelled_by_all_integrators[crossed]
            selected = np.argmax(dist_by_winners)
            decision = crossed[selected]
        elif not broken:
            decision = crossed[0]


        if self.over_actions:
            action = decision
        elif not broken:
            action = controls[decision]
        else:
            action = -1
            broken = False

        self.RT[tau,t] = i


        if False:
            self.episode_walks.append(decision_log[:i+1,:])
            plt.plot(np.arange(0, decision_log[:i+1,:]), decision_log[:i+1,:])
            plt.show()
            if(t == self.T-2):
                self.walks.append(self.episode_walks.copy())
                self.episode_walks = []
            elif self.T == 2:
                self.walks.append(decision_log[:i+1,:])
        
        return action


 
'''
 _______   _______   __       __ 
/       \ /       \ /  \     /  |
$$$$$$$  |$$$$$$$  |$$  \   /$$ |
$$ |__$$ |$$ |  $$ |$$$  \ /$$$ |
$$    $$< $$ |  $$ |$$$$  /$$$$ |
$$$$$$$  |$$ |  $$ |$$ $$ $$/$$ |
$$ |  $$ |$$ |__$$ |$$ |$$$/ $$ |
$$ |  $$ |$$    $$/ $$ | $/  $$ |
$$/   $$/ $$$$$$$/  $$/      $$/ 
'''
                                 
                            


class RacingDiffusionSelector(object):


    def __init__ (self, trials, T, number_of_actions=2, wd = 1, s = 1, b = 1, A = 1, v0 = 0, over_actions = False):

        
        self.s  = np.sqrt(s)      # standard deviation of decision variable
        self.wd = wd              # weight of advantage term
        self.v0 = v0              # urgency / bias term
        self.b = b                # boundary, needs to be implemented
        self.A = A                # range of possible starting points

        self.type = 'rdm'
        self.trials = trials
        self.T = T
        self.RT = np.zeros([trials,T-1])
        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
        self.walks = []
        self.episode_walks = []
        self.over_actions = over_actions
        self.prior_as_starting_point = True
        self.sample_other = False
        self.sample_posterior = False

    def reset_beliefs(self):
        pass
        # self.control_probability[:,:,:] = 0


    def set_pars(self, pars):
        pass    


    def log_prior(self):
        return 0


    
    def estimate_action_probability(self, tau, dist_policies, actions, *args):

        #estimate action probability
        na = np.unique(actions).size
        control_prob = np.zeros(na)

        for a in range(na):
            control_prob[a] = dist_policies[actions == a].sum()

    #    normalize?

        # print(control_prob)
        # print(control_prob /control_prob.sum())
        return  control_prob 

    def select_desired_action(self,tau, t, posterior, controls, *args): #avg_likelihood, prior, dt = 0.01, plot=False):
   
        likelihood = args[0]
        prior = args[1]


        if (len(args) == 3):
            dt = args[3]
        else: 
            dt = 0.01
    
        if self.over_actions:
            prior = self.estimate_action_probability(tau, prior, controls)
            likelihood = self.estimate_action_probability(tau, likelihood, controls)
            posterior = self.estimate_action_probability(tau, posterior, controls)

        ni = prior.size                                            # number of integrators, = either na or npi
        decision_log = np.zeros([10005,ni])                        # log for propagated decision variable

        if self.prior_as_starting_point:
            decision_log[0,:] = self.A*prior                        # initial conditions for v_ij

        v0 = np.ones(ni)*self.v0                                    # vectorized urgency term
        bound_reached = np.zeros(ni, dtype = bool)
        i = 0

        if self.sample_posterior:
            Q = posterior
        elif self.sample_other:
            Q  = likelihood+prior
        else: 
            Q = likelihood

        broken = False
        while(not np.any(bound_reached)):                        # if still no decision made
            dW = np.random.normal(scale = self.s, size = ni)
            decision_log[i+1,:] = decision_log[i,:] + (v0 + self.wd*Q)*dt + (Q>0)*dW
            bound_reached = decision_log[i+1,:] >= self.b
            i+=1

            if i > 10000:
                action = -1
                broken = True
                break

        crossed = bound_reached.nonzero()[0]                      # non-zero returns tuple by default
        
        
        if (crossed.size > 1):
            vals = decision_log[i,crossed]
            selected = np.argmax(vals)
            decision = crossed[selected]

            # raise ValueError('Two integrators crossed decision boundary at the same time')
            # print('Two integrators crossed boundary at the same time, choosing one at random')
            decision = np.random.choice(crossed, p=np.ones(crossed.size)/crossed.size)
        elif not broken:
            decision = crossed[0]

        if self.over_actions:
            action = decision
        elif not broken:
            action = controls[decision]
        else:
            action = -1
            broken = False

        self.RT[tau,t] = i




        if False:
            self.episode_walks.append(decision_log[:i+1,:])
            if(t == self.T-2):
                self.walks.append(self.episode_walks.copy())
                self.episode_walks = []
            elif self.T == 2:
                self.walks.append(decision_log[:i+1,:])
        
        return action


'''
fit it with grid world - look for decrease
artificial script with 2/3 choices and various prior/like combos
fitting of two stage task if 2/3 choices work
flanker
task switching

'''

class MCMCSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, ESS = 50):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
        self.ess = ESS
        self.RT = np.zeros((trials, T-1))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        npi = posterior_policies.shape[0]
        likelihood = args[0]
        prior = args[1]
        accepted_pis = np.zeros(50000, dtype=np.int32) - 1

        curr_ess = 0
        i = 0

        pi = np.random.choice(npi, p=prior)
        accepted_pis[i] = pi
        i += 1
        while (curr_ess < self.ess) or (i<10*self.ess):

            pi = np.random.choice(npi, p=prior)
            r = np.random.rand()
            #print(i, curr_ess)

            if likelihood[pi]/likelihood[accepted_pis[i-1]] > r:#posterior_policies[pi]/posterior_policies[accepted_pis[i-1]] > r:
                accepted_pis[i] = pi
            else:
                accepted_pis[i] = accepted_pis[i-1]

            autocorr = acov(accepted_pis[:i+1])
            #print(autocorr)

            if autocorr[0] > 0:
                ACT = 1 + 2*np.abs(autocorr[1:]).sum()/autocorr[0]
                curr_ess = i/ACT
            else:
                ACT = 0
                curr_ess = 1

            i += 1

        self.RT[tau,t] = i-1
        print(tau, t, i-1)

        u = actions[accepted_pis[i-1]]

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class DirichletSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, factor=0.4, calc_dkl=False, calc_entropy=False, draw_true_post=False):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
        self.RT = np.zeros((trials, T-1))
        self.factor = factor
        self.draw_true_post = draw_true_post

        self.calc_dkl = calc_dkl
        if calc_dkl:
            self.DKL_post = np.zeros((trials, T-1))
            self.DKL_prior = np.zeros((trials, T-1))
        self.calc_entropy = calc_entropy
        if calc_entropy:
            self.entropy_post = np.zeros((trials, T-1))
            self.entropy_prior = np.zeros((trials, T-1))
            self.entropy_like = np.zeros((trials, T-1))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        npi = posterior_policies.shape[0]
        likelihood = args[0]
        prior = args[1] #np.ones_like(likelihood)/npi #
        # likelihood = np.array([0.5,0.5])
        # prior = np.array([0.5,0.5])
        # posterior_policies = prior * likelihood
        # posterior_policies /= posterior_policies.sum()
        #print(posterior_policies, prior, likelihood)
        self.accepted_pis = np.zeros(100000, dtype=np.int32) - 1
        dir_counts = np.ones(npi, np.double)

        curr_ess = 0
        i = 0

        H_0 =         + (dir_counts.sum()-npi)*scs.digamma(dir_counts.sum()) \
                        - ((dir_counts - 1)*scs.digamma(dir_counts)).sum() \
                        + logBeta(dir_counts)
        #print("H", H_0)

        pi = np.random.choice(npi, p=prior)
        self.accepted_pis[i] = pi
        dir_counts[pi] += 1
        H_dir =         + (dir_counts.sum()-npi)*scs.digamma(dir_counts.sum()) \
                        - ((dir_counts - 1)*scs.digamma(dir_counts)).sum() \
                        + logBeta(dir_counts)
        #print("H", H_dir)

        if t == 0:
            i += 1
            while H_dir>H_0 - self.factor + self.factor*H_0:

                pi = np.random.choice(npi, p=prior)
                r = np.random.rand()
                #print(i, curr_ess)

                #acc_prob = min(1, posterior_policies[pi]/posterior_policies[self.accepted_pis[i-1]])
                if likelihood[self.accepted_pis[i-1]]>0:
                    acc_prob = min(1, likelihood[pi]/likelihood[self.accepted_pis[i-1]])
                else:
                    acc_prob = 1
                if acc_prob >= r:#posterior_policies[pi]/posterior_policies[self.accepted_pis[i-1]] > r:
                    self.accepted_pis[i] = pi
                    dir_counts[pi] += 1#acc_prob
                else:
                    self.accepted_pis[i] = self.accepted_pis[i-1]
                    dir_counts[self.accepted_pis[i-1]] += 1#1-acc_prob

                H_dir =     + (dir_counts.sum()-npi)*scs.digamma(dir_counts.sum()) \
                            - ((dir_counts - 1)*scs.digamma(dir_counts)).sum() \
                            + logBeta(dir_counts)
                #print("H", H_dir)

                i += 1

            self.RT[tau,t] = i-1
            #print(tau, t, i-1)
        else:
            self.RT[tau,t] = 0

        if self.draw_true_post:
            chosen_pol = np.random.choice(npi, p=posterior_policies)
        else:
            chosen_pol = self.accepted_pis[i-1]

        u = actions[chosen_pol]
        #print(tau,t,iself.accepted_pis[i-1],u,H_rel)
        # if tau in range(100,110) and t==0:
        #     plt.figure()
        #     plt.plot(posterior_policies)
        #     plt.show()

        if self.calc_dkl:
            # autocorr = acov(self.accepted_pis[:i+1])

            # if autocorr[0] > 0:
            #     ACT = 1 + 2*np.abs(autocorr[1:]).sum()/autocorr[0]
            #     ess = i/ACT
            #     ess = round(ess)
            # else:
            #     ess = 1

            dist = dir_counts / dir_counts.sum()
            D_KL = entropy(posterior_policies, dist)
            self.DKL_post[tau,t] = D_KL
            D_KL = entropy(prior, dist)
            self.DKL_prior[tau,t] = D_KL

        if self.calc_entropy:
            self.entropy_post[tau,t] = entropy(posterior_policies)
            self.entropy_prior[tau,t] = entropy(prior)
            self.entropy_like[tau,t] = entropy(likelihood)
            # if t==0:
            #     print(tau)
            #     n = 12
            #     ind = np.argpartition(posterior_policies, -n)[-n:]
            #     print(np.sort(ind))
            #     print(np.sort(posterior_policies[ind]))

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class DKLSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2, ESS = 50):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))
        self.ess = ESS
        self.RT = np.zeros((trials, T-1))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        npi = posterior_policies.shape[0]
        likelihood = args[0]
        prior = args[1]

        DKL = (likelihood * ln(likelihood/prior)).sum()
        H = - (posterior_policies * ln(posterior_policies)).sum()
        H_p = - (prior * ln(prior)).sum()

        self.RT[tau,t] = np.exp(H_p + np.random.normal(H, DKL))

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)
        u = np.random.choice(self.na, p = self.control_probability[tau, t])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class AveragedSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        #generate the desired response from action probability
        u = np.random.choice(self.na, p = self.control_probability[tau, t])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()


        self.control_probability[tau, t] = control_prob


class MaxSelector(object):

    def __init__(self, trials = 1, T = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions
        self.control_probability = np.zeros((trials, T, self.na))

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        self.estimate_action_probability(tau, t, posterior_policies, actions)

        #generate the desired response from maximum policy probability
        indices = np.where(posterior_policies == np.amax(posterior_policies)) # same as np.argmax but selects all cases wher the max element appears
        u = np.random.choice(actions[indices])

        return u

    def estimate_action_probability(self, tau, t, posterior_policies, actions, *args):

        #estimate action probability
        control_prob = np.zeros(self.na)
        for a in range(self.na):
            control_prob[a] = posterior_policies[actions == a].sum()

        self.control_probability[tau, t] = control_prob


class AveragedPolicySelector(object):

    def __init__(self, trials = 1, T = 10, number_of_policies = 10, number_of_actions = 2):
        self.n_pars = 0

        self.na = number_of_actions

        self.npi = number_of_policies

    def reset_beliefs(self):
        self.control_probability[:,:,:] = 0

    def set_pars(self, pars):
        pass

    def log_prior(self):
        return 0

    def select_desired_action(self, tau, t, posterior_policies, actions, *args):

        #generate the desired response from policy probability
        npi = posterior_policies.shape[0]
        pi = np.random.choice(npi, p = posterior_policies)

        u = actions[pi]

        return u
