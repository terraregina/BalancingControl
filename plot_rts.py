from misc import calc_dkl, extract_params_from_ttl
import pickle as pickle 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import itertools as itertools
from misc import run_action_selection, test_vals, params_dict
from misc import load_data, load_data_from_ttl
import os 
import pandas as pd 
from scipy import stats


''' PARAMETERS'''
methods = ['rdm','ardm']
modes = ['conflict', 'agreement', 'goal', 'habit']
nmodes = len(modes)
trials = 1000
polss =  np.array([3,8,81])
nmodes = len(modes)

params_list = [[False, False, True, 'standard'],\
              [True, False, True, 'post_prior1' ],\
              [True, False, False, 'post_prior0'],\
              [False, True, True, 'like_prior1' ],\
              [False, True, False, 'like_prior0']]

path = os.getcwd() + '\\parameter_data\\'

parameter_names = ['npi', 'methods', 'b', 'wd', 's', 'params_list']


'''
BRUTE FORCE PARAMETER SEARCH
'''
pols = np.array([3]) #,8,81]
par_list = []
parameter_names = ['npi', 'methods', 'b', 'wd', 's', 'params_list']
trials = 1000

# ss = [0.01, 0.03, 0.05, 0.07, 0.1]
# wds = np.arange(0.5,2.5,0.5)
# bs = np.arange(1, 2.5, 0.5)

# ss = [0.01, 0.03, 0.05, 0.07, 0.1]
# wds = [0.7, 0.9, 1.3, 1.7, 1.9, 2.1, 2.3]
# bs = [1, 1.3, 1.7, 1.9, 2.1, 2.3]

bs = np.arange(1,3,0.3).round(4)
bs = np.arange(1,3,0.3).round(4)

ss = np.arange(0.005, 0.011, 0.001).round(5)
wds = np.arange(200, 10, -10)
size = wds.size+1
wds = wds[np.newaxis,:]*ss[:,np.newaxis]
wds = wds + ss[:,np.newaxis]
drift_var = np.column_stack((wds.ravel(), ss.repeat(size-1))).round(6)
# print(drift_var)
# print((0.01*0.5*drift_var[:,0] + drift_var[:,1]).reshape(len(ss), size-1))



for p in itertools.product(pols, methods, bs, drift_var, params_list):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]])


def generate_data():

    # not currently iterating over policy sizes
    for ind, p in enumerate(par_list):
        print(ind)
        npi = p[0]
        selector = p[1]
        b = p[2]
        wd = p[3][0]
        s = p[3][1]
        sample_post, sample_other, prior_as_start, reg = p[4]

        ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 
        print('\n' + ttl)
        empirical = np.zeros([nmodes, npi])
        RT = np.zeros([nmodes, trials])

        for m, mode in enumerate(modes):
            i = np.where(polss == npi)[0][0]
            prior = test_vals[i][m][1]
            like = test_vals[i][m][2]
            post = test_vals[i][m][0]
            # print('variance:', s)
            actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                            prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
                            var=s, wd=wd, b=b)
            actions = np.asarray(actions)
            actions = actions[actions != -1]
            actions = actions.tolist()
            empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
            RT[m,:] = ac_sel.RT.squeeze()
            

        
        ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 

        with open(path + ttl, 'wb') as fp:

            dict = {
                'RT': RT,
                'empirical': empirical,
                'parameters': parameter_names,
                'parameter_values':p
            }

            pickle.dump(dict, fp)

        # if False:
        #     with open(path + ttl, 'rb') as fp:
        #         test = pickle.load(fp)

        #     for val in test:
        #         print(val)

'''
EVOLUTIONARY SEARCH
'''

def calc_penalty(fit, agr_median, desired_median = 300, m=100, k=0.165, alpha=0, beta=1):
    
    median_penalty = ((agr_median - desired_median)/m)**2 + 1
    post_fit_penalty = np.exp(fit/k)

    return [alpha*median_penalty + beta*post_fit_penalty, median_penalty, post_fit_penalty]

def find_best_params(initial_params, sd_b= 0.01, sd_wd = 10, no = 2, iters=1000,tol= 1e-3):
    npi = initial_params[0]
    selector = initial_params[1]
    b = initial_params[2]
    wd = initial_params[3][0]
    s = initial_params[3][1]
    sample_post, sample_other, prior_as_start, reg = initial_params[4]
    
    ind = np.where(polss == npi)[0][0]
    posts = np.zeros([len(modes), 3])    # translate the posteriors
    post = np.asarray(test_vals)[ind,:,0]  # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    best_fit = np.infty
    i = 0

    post_fits = np.zeros(no+1)
    fit_penalties = np.zeros(no+1)
    agr_meds = np.zeros(no+1)
    med_penalties = np.zeros(no+1)
    penalties = np.zeros(no+1)

    # calc fitness of initial guess
    empirical = np.zeros([nmodes, npi])
    RT = np.zeros([nmodes, trials])

    for m, mode in enumerate(modes):
        prior = test_vals[ind][m][1]
        like = test_vals[ind][m][2]
        post = test_vals[ind][m][0]

        actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                        prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
                        var=s, wd=wd, b=b)

        actions = np.asarray(actions)
        actions = actions[actions != -1]
        actions = actions.tolist()
        empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
        RT[m,:] = ac_sel.RT.squeeze()

    post_fits[0] = np.abs((posts - empirical)/posts).mean(axis=1).mean()
    agr_meds[0] = np.median(RT[1,:])
    penalties[0], med_penalties[0], fit_penalties[0] = calc_penalty(post_fits[0], agr_meds[0])

    while(best_fit > tol or i < iters):

        # generate offspring
        # bs = np.append(np.array([b]), np.random.normal(b,sd_b,no))
        bs = np.append(np.array([b]), np.random.normal(b,0,no))
        # wds = np.append(np.array([wds]), np.random.normal(wd, wd*0.05, no))
        wds = (wd/s+ np.append(np.array([0]), np.random.normal(0,1.5, no)))*s
        # ss = np.append(np.array([s]), np.random.normal(s, s*0.05, no))
        ss = np.append(np.array([s]), np.random.normal(s, 0, no))
     
        # calc offspring fitness
        for o in range(no+1):
            empirical = np.zeros([nmodes, npi])
            RT = np.zeros([nmodes, trials])

            for m, mode in enumerate(modes):
                prior = test_vals[ind][m][1]
                like = test_vals[ind][m][2]
                post = test_vals[ind][m][0]

                actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                                prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
                                var=ss[o], wd=wds[o], b=bs[o])

                actions = np.asarray(actions)
                actions = actions[actions != -1]
                actions = actions.tolist()
                empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
                RT[m,:] = ac_sel.RT.squeeze()

            post_fits[o] = np.abs((posts - empirical)/posts).mean(axis=1).mean()
            agr_meds[o] = np.median(RT[1,:])
            penalties[o], med_penalties[o], fit_penalties[o] = calc_penalty(post_fits[o], agr_meds[o])

        # select best candidate
        
        print("\npost_fits", post_fits)
        print("fit_penalties", fit_penalties)
        print("med_penalties", med_penalties)
        print("penalties", penalties)
        best = np.argmin(penalties)
        b, bs[best] = [bs[best]]*2
        wd, wds[best] = [wds[best]]*2
        s, ss[best] = [ss[best]]*2
        post_fits[0] = post_fits[best]
        fit_penalties[0] = fit_penalties[best]
        agr_meds[0] = agr_meds[best]
        med_penalties[0] = med_penalties[best]
        penalties[0] = penalties[best]
        best_fit = post_fits[best]
        print(i, best, best_fit)
        print(b,wd)

        i += 1

def sim_plot_rt_post(initial_params, trials = 2000):

    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    npi = initial_params[0]
    selector = initial_params[1]
    b = initial_params[2]
    wd = initial_params[3][0]
    s = initial_params[3][1]
    sample_post, sample_other, prior_as_start, reg = initial_params[4]

    empirical = np.zeros([nmodes, npi])
    RT = np.zeros([nmodes, trials])
    
    ind = np.where(polss == npi)[0][0]
    
    x_positions = []
    for i in range(4):
        x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])
    
    fig, ax = plt.subplots(2,1)

    for m, mode in enumerate(modes):
        prior = test_vals[ind][m][1]
        like = test_vals[ind][m][2]
        post = test_vals[ind][m][0]

        actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                        prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
                        var=s, wd=wd, b=b)


        height = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
        empirical[m,:] =height
        x_pos = x_positions[m]
        lab =' '.join([mode, 'mode',  str(stats.mode(ac_sel.RT)[0][0][0]), 'median', str(np.median(ac_sel.RT)), 'mean', str(ac_sel.RT.mean())])
        ax[0].hist(ac_sel.RT, bins=100, alpha=0.5, label=lab)
        if m == 0:
            ax[1].bar(x_pos, post, alpha=0.5, color='k', label = "post" )
        else:
                ax[1].bar(x_pos, post, alpha=0.5, color='k')

        ax[1].bar(x_pos, height, label=mode + ' empir', alpha=0.5, color=cols[m])

    ax[0].legend()
    ax[1].legend()
    posts = np.zeros([len(modes), 3])    # translate the posteriors
    post = np.asarray(test_vals)[ind,:,0]  # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)

    plt.show()
    print( np.abs((posts - empirical)/posts).mean(axis=1).mean())



####################################################
####################################################
####################################################

'''
OPTIMIZE
'''

regime = 'like_prior0'
initial_params = [3, 'ardm', 1.0061389130038418, [1.3957648731779595, 0.00885223], params_dict[regime] + [regime]]
regime = 'like_prior0'
# initial_params = [3, 'rdm', 1.55336596842844, [ 1.5108917222399416, 0.00456143], params_dict[regime] + [regime]]
s =  0.000456143*0.7
initial_params = [3, 'rdm', 1, [ 330*s, s], params_dict[regime] + [regime]]
# find_best_params(initial_params)



'''
SIMULATE AND PLOT RT DIST WITH POST APPROX
'''
    
# sim_plot_rt_post(initial_params)
# print('poop')



'''
EXPLORE AROUND TOP CANDIDATES
'''

df = load_data()
# print the fit values for all the best fits
df['opt_fit_group'] = df.groupby(['npi', 'selector','regime'])['fit'].transform('min')
df['optimal'] =(df['fit'] == df['opt_fit_group'])*1
# print(df[df['optimal'] == 1])
# best = df[(df['regime']=='like_prior1') & (df['optimal']==1) & (df['selector']=='ardm')]
'''''''''''''''''''''''
'''''''''''''''''''''''
# print(best)

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
trials = 1000

posts = np.zeros([len(modes), 3])      # translate the posteriors
post = np.asarray(test_vals)[0,:,0]    # into a numpy array

for indx, p in enumerate(post):
    posts[indx,:] = np.asarray(p)

for ind, row in df[df['optimal'] == 1].iterrows():
    npi, selector, pars, regime, s, wd, b = extract_params_from_ttl(row.title)
    ratio = wd/s
    # wds = np.arange(-10,10,1)
    # wds = wds[wds != 0]
    wds = np.arange(0,3)
    wds = (wds + ratio)*s
    sample_post, sample_other, prior_as_start = params_dict[regime]
    
    for ind, wd in enumerate(wds): 
        ttl = '_'.join(['npi', str(npi), selector, regime, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 
        print('\n' + ttl)
        empirical = np.zeros([nmodes, npi])
        RT = np.zeros([nmodes, trials])

        for m, mode in enumerate(modes):
            i = np.where(polss == npi)[0][0]
            prior = test_vals[i][m][1]
            like = test_vals[i][m][2]
            post = test_vals[i][m][0]
            # print('variance:', s)
            actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                            prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
                            var=s, wd=wd, b=b)
            actions = np.asarray(actions)
            actions = actions[actions != -1]
            actions = actions.tolist()
            empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
            RT[m,:] = ac_sel.RT.squeeze()
            
        post_fit = np.abs((posts - empirical)/posts).mean(axis=1).mean()
        print(post_fit)
        ttl = '_'.join(['npi', str(npi), selector, regime, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 

        with open(path + ttl, 'wb') as fp:

            dict = {
                'RT': RT,
                'empirical': empirical,
                'parameters': parameter_names,
                'parameter_values':p
            }

            pickle.dump(dict, fp)




# generate_data()


'''
PLOT APPROXIMATION FROM DATA FILE AND STATS
'''

ttl = 'npi_3_rdm_standard_b_1_wd_0.025_s_2.5e-05_.txt'
def plot_from_file(ttl):
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = plt.subplot(211)
    ax2 = plt.subplot(212)
    npi, selector, pars, regime, s, wd, b = extract_params_from_ttl(row.title)
    
    with open(path + ttl, 'rb') as fp:
        data = pickle.load(fp)
    
    x_positions = []
    for i in range(4):
        x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

    empirical = np.asarray(data['empirical'])
    RT = np.asarray(data['RT'])
    df = pd.DataFrame(RT.T, index = np.arange(RT.shape[1]), columns=modes)
    print(df.describe())
    print(df.mode())
    print(df.median())

    for i in range(4):
        x_pos = x_positions[i]
        post = test_vals[0][i][0]
        ax.bar(x_pos, post, label='post', alpha=0.5, color="k")
        ax.bar(x_pos, empirical[i,:], label='empirical', alpha=0.5, color=cols[i])
        ax2.hist(RT[i,:], bins=100, alpha=0.5)
    plt.show()

