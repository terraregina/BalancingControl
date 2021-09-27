import numpy as np
import matplotlib.pyplot as plt
import action_selection

'''DEFINE DUMMY DISTRIBUTIONS'''

tests = ["conflict", "agreement", "goal", "habit"]
num_tests = len(tests)
test_vals = [[],[],[]]

# ///////// setup 2 policies

npi = 3
gp = 1
high_prob_1 = 0.7
high_prob_2 = 0.7
flat = np.ones(npi)/npi
h1 = np.array([high_prob_1]*gp + [(1 - high_prob_1*gp)/(npi-gp)]*(npi-gp))
h2_conf = np.array([(1 - high_prob_2*gp)/(npi-gp)]*gp + [high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp*2))
h2_agree = np.array([high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp))

# conflict
prior = h1.copy()
like = h2_conf.copy()
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# agreement
prior = h1.copy()
like = h2_agree.copy()
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

#goal
prior = flat
like = h1.copy()
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])

# habit
prior = h1.copy()
like = flat
post = prior*like
post /= post.sum()
test_vals[0].append([post,prior,like])


############# setup 8 policies

# ///////// setup 8 policies

npi = 8
gp = 2
high_prob_1 = 0.4
high_prob_2 = 0.3
flat = np.ones(npi)/npi
h1 = np.array([high_prob_1]*gp + [(1 - high_prob_1*gp)/(npi-gp)]*(npi-gp))
h2_conf = np.array([(1 - high_prob_2*gp)/(npi-gp)]*gp + [high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp*2))
h2_agree = np.array([high_prob_2]*gp + [(1 - high_prob_2*gp)/(npi-gp)]*(npi-gp))

# conflict
prior = h1.copy()
like = h2_conf.copy()
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])

# agreement
prior = h1.copy()
like = h2_agree.copy()
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])

#goal
prior = flat
like = h1.copy()
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])

# habit
prior = h1.copy()
like = flat
post = prior*like
post /= post.sum()
test_vals[1].append([post,prior,like])


############# setup 81 policies
num_tests = len(tests)
gp = 6
n = 81
val = 0.148
l = [val]*gp+[(1-6*val)/(n-gp)]*(n-gp)
v = 0.00571
p = [(1-(v*(n-gp)))/6]*gp+[v]*(n-gp)
conflict = [v]*gp+[(1-(v*(n-gp)))/6]*gp+[v]*(n-2*gp)
npi = n
flat = [1./npi]*npi

# conflict
prior = np.array(conflict)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])

# agreement
prior = np.array(p)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])

# goal
prior = np.array(flat)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])

# habit
prior = np.array(l)
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals[2].append([post,prior,like])


plot = False

if plot:
    for y in range(2):
        # print(y)
        fig, ax = plt.subplots(2,2, figsize = (8,6))
        ax = ax.reshape(ax.size)
        for i in range(4):
            ax[i].plot(test_vals[y][i][1], label='prior')
            ax[i].plot(test_vals[y][i][2], label='likelihood')
            ax[i].plot(test_vals[y][i][0], label='posterior')
            ax[i].set_title(tests[i])
            plt.legend()
        plt.show()


'''Calculate how posterior is approximated for different setups dependent on variance.
Hence I will have a 2 x 4 figure where the top panel will have the engineered dists and
below there will be DKL vs variance. I need to generate this for all variants.
'''


def run_action_selection(selector, prior, like, post, trials=10, T=2, prior_as_start=True, sample_post=False, sample_other=False):
    
    na = prior.shape[0]
    controls = np.arange(0, na, 1)
    if selector == 'rdm':
        # not really over_actions, simply avoids passing controls
        ac_sel = action_selection.RacingDiffusionSelector(trials, T, number_of_actions=na, s=var, over_actions=False)
    elif selector == 'ardm':
        # not really over_actions, simply avoids passing controls
        ac_sel = action_selection.AdvantageRacingDiffusionSelector(trials, T, number_of_actions=na, s=var, over_actions=False)
    else: 
        raise ValueError('Wrong or no action selection method passed')
    
    ac_sel.prior_as_starting_point = prior_as_start
    ac_sel.sample_other = sample_other
    ac_sel.sample_posterior = sample_post
    print('prior as start, sample_other, sample_post')
    print(ac_sel.prior_as_starting_point, ac_sel.sample_other, ac_sel.sample_posterior)
    actions = []
    
    for trial in range(trials):
        actions.append(ac_sel.select_desired_action(trial, 0, post, controls, like, prior))   #trial, t, post, control, like, prior


    return actions, ac_sel


def calc_dkl(p,q):
    # print(p)
    # print(q)
    p[p == 0] = 10**(-300)
    q[q == 0] = 10**(-300)

    ln = np.log(p/q)
    if np.isnan(p.dot(ln)):
        raise ValueError('is Nan')

    return p.dot(ln)


###########################################

pols = [3,8,81]
vars = np.arange(0.01,0.11,0.01)
trials = 1000
method = 'ardm'

for i, npi in enumerate(pols):
    dkl = np.zeros([len(pols), len(tests), len(vars)])
    fig, axes = plt.subplots(2,4, figsize=(15,5))
    axes[0,0].set_ylim([0,1])
    for r, regime in enumerate(tests):
        for s, var in enumerate(vars):
            prior = test_vals[i][r][1] 
            like = test_vals[i][r][2]
            post = test_vals[i][r][0]
            
            actions, selector = run_action_selection(method, prior, like, post, trials=trials)

            empirical = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
            dkl[i,r,s] = calc_dkl(empirical, prior)
            
        axes[0][r].plot(prior, label='prior')
        axes[0][r].plot(like, label='like')
        axes[0][r].plot(post, label='post')
        axes[0][r].legend()
        axes[0][r].set_title(tests[r])

        axes[1][r].plot(vars, dkl[i,r,:],'-o')

    plt.setp(axes, ylim=axes[0,0].get_ylim())
    fig.suptitle(method + ', ' + str(trials) + ' trials, likelihood as drift rate, DKL of empirical relative to posterior')        
    ttl = 'dkl_'+method+'_npi'+str(npi)+'_standard.png'
    # plt.show()
    plt.savefig(ttl)
    plt.close()


        #  post, like , prior  
params_list = [[False, True, True], [False, True, False]]#[[True, False, False], [True, False, True], [False, True, False], [False, True, True]]
regimes = ['posterior, no starting point', 'posterior with prior as starting point', 'like+prior, no starting point', 'like+prior, prior as starting point']

for p, pars in enumerate(params_list):

    for i, npi in enumerate(pols):
        dkl = np.zeros([len(pols), len(tests), len(vars)])
        fig, axes = plt.subplots(2,4, figsize=(15,5))
        axes[0,0].set_ylim([0,1])
        for r, regime in enumerate(tests):
            for s, var in enumerate(vars):
                prior = test_vals[i][r][1] 
                like = test_vals[i][r][2]
                post = test_vals[i][r][0]
                
                actions, selector = run_action_selection(method, prior, like, post, trials=trials,\
                                                          sample_post=pars[0], sample_other=pars[1], prior_as_start=pars[2])

                empirical = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
                dkl[i,r,s] = calc_dkl(empirical, prior)
                
            axes[0][r].plot(prior, label='prior')
            axes[0][r].plot(like, label='like')
            axes[0][r].plot(post, label='post')
            axes[0][r].legend()
            axes[0][r].set_title(tests[r])

            axes[1][r].plot(vars, dkl[i,r,:],'-o')

        plt.setp(axes, ylim=axes[0,0].get_ylim())
        fig.suptitle(method + ', ' + str(trials) + ' trials, ' + regimes[p] + ', DKL relative to posterior')        
        ttl = 'dkl_' + method +'_npi'+str(npi)+'_'+ regimes[p]+'.png'
        # plt.show()
        plt.savefig(ttl)