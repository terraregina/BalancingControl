# %%
from misc import calc_dkl, extract_params_from_ttl
import pickle as pickle 
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
import itertools as itertools
from misc import run_action_selection, test_vals, params_dict
from misc import load_data, load_data_from_ttl, simulate, make_title
import os 
import pandas as pd 
from scipy import stats


''' PARAMETERS'''
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
methods = ['ardm','rdm']
modes = ['conflict', 'agreement', 'goal', 'habit']
nmodes = len(modes)
trials = 1000
polss =  np.array([3,8,81,2])
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


# def simulate(i, selector, b, s, wd, sample_post, sample_other, prior_as_start, plot=False, calc_fit=False):
#     npi=2
#     empirical = np.zeros([nmodes, npi])
#     RT = np.zeros([nmodes, trials])

#     if plot:
#         x_positions = []
#         for i in range(4):
#             x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

#         fig, ax = plt.subplots(2,1)

#     for m, mode in enumerate(modes):
#         i = np.where(polss == npi)[0][0]
#         prior = test_vals[i][m][1]
#         like = test_vals[i][m][2]
#         post = test_vals[i][m][0]
#         # print('variance:', s)
#         actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
#                         prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
#                         var=s, wd=wd, b=b)

#         actions = np.asarray(actions)
#         actions = actions[actions != -1]
#         actions = actions.tolist()
#         empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
#         RT[m,:] = ac_sel.RT.squeeze()

#         if plot:
#             x_pos = x_positions[m]
#             lab =' '.join([mode, 'mode',  str(stats.mode(ac_sel.RT)[0][0][0]), 'median', str(np.median(ac_sel.RT)), 'mean', str(ac_sel.RT.mean())])
#             ax[0].hist(ac_sel.RT, bins=100, alpha=0.5, label=lab)
            
#             if m == 0:
#                 ax[1].bar(x_pos, post, alpha=0.5, color='k', label = "post" )
#             else:
#                 ax[1].bar(x_pos, post, alpha=0.5, color='k')

#             ax[1].bar(x_pos, empirical[m,:], label=mode + ' empir', alpha=0.5, color=cols[m])

#     if plot:
#         ax[0].legend()
#         ax[1].legend()
#         plt.show()

#     if calc_fit:
#         posts = np.zeros([len(modes), 3])    # translate the posteriors
#         post = np.asarray(test_vals)[i,:,0]  # into a numpy array

#         for indx, p in enumerate(post):
#             posts[indx,:] = np.asarray(p)

#             fit =  np.abs((posts - empirical)/posts).mean(axis=1).mean()

#     if calc_fit:
#         return actions, empirical, RT, fit
#     else:
#         return actions, empirical, RT

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
        i = np.where(polss == npi)[0][0]
        actions, empirical, RT = simulate(selector, b,s,wd, sample_post, sample_other, prior_as_start, npi=npi)

        # empirical = np.zeros([nmodes, npi])
        # RT = np.zeros([nmodes, trials])

        # for m, mode in enumerate(modes):
        #     i = np.where(polss == npi)[0][0]
        #     prior = test_vals[i][m][1]
        #     like = test_vals[i][m][2]
        #     post = test_vals[i][m][0]
        #     # print('variance:', s)
        #     actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
        #                     prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
        #                     var=s, wd=wd, b=b)
        #     actions = np.asarray(actions)
        #     actions = actions[actions != -1]
        #     actions = actions.tolist()
        #     empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
        #     RT[m,:] = ac_sel.RT.squeeze()
            
        
        ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 

        with open(path + ttl, 'wb') as fp:

            dict = {
                'RT': RT,
                'empirical': empirical,
                'parameters': parameter_names,
                'parameter_values':p
            }

            pickle.dump(dict, fp)


def calc_penalty(fit, agr_median, desired_median = 300, m=100, k=0.165, alpha=0, beta=1):
    
    median_penalty = ((agr_median - desired_median)/m)**2 + 1
    post_fit_penalty = np.exp(fit/k)

    return [alpha*median_penalty + beta*post_fit_penalty, median_penalty, post_fit_penalty]


def find_best_params(initial_params, sd_b= 0.01, sd_wd = 10, no = 2, iters=100,tol= 1e-3):
    npi = initial_params[0]
    selector = initial_params[1]
    b = initial_params[2]
    wd = initial_params[3][0]
    s = initial_params[3][1]
    sample_post, sample_other, prior_as_start, reg = initial_params[4]
    
    ind = np.where(polss == npi)[0][0]
    posts = np.zeros([len(modes), npi])      # translate the posteriors
    post = np.asarray(test_vals)[ind,:,0]  # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    best_fit = np.infty
    i = 0

    post_fits = np.zeros([trials, no+1])
    fit_penalties = np.zeros([trials, no+1])
    agr_meds = np.zeros([trials, no+1])
    med_penalties = np.zeros([trials, no+1])
    penalties = np.zeros([trials, no+1])
    best_wds = np.zeros(trials)
    # calc fitness of initial guess
    np.random.seed(0)
    action, empirical, RT = simulate(selector, b, s, wd, sample_post, sample_other, prior_as_start, npi=npi)

    post_fits[i,0] = np.abs((posts - empirical)/posts).mean(axis=1).mean()
    agr_meds[i,0] = np.median(RT[1,:])
    penalties[i,0], med_penalties[i,0], fit_penalties[i,0] = calc_penalty(post_fits[i,0], agr_meds[i,0])
 
    best_fit_counter = 0
    total = 0
    last_best_fit = 1042342

    factor = 0.15
    while(best_fit > tol and i < iters and total < 30):

        np.random.seed()
        # generate offspring
        # bs = np.append(np.array([b]), np.random.normal(b,sd_b,no))
        # wds = np.append(np.array([wds]), np.random.normal(wd, wd*0.05, no))
        
        # if (best_fit< 0.05):
        #     factor = 0.002
        # elif(best_fit< 0.1):
        #     factor = 0.01
        # elif(best_fit < 0.3):
        #     factor = 0.1
        # else:

        bs = np.append(np.array([b]), np.random.normal(b,0,no))
        wds = (wd/s+ np.append(np.array([0]), np.random.normal(0,wd/s*factor, no)))*s
        # wds = np.append(np.array([wd]), np.random.normal(wd, 0, no)) 
        print("wds", wds)
        # ss = np.append(np.array([s]), np.random.normal(s, s*0.05, no))
        ss = np.append(np.array([s]), np.random.normal(s, 0, no))
     
        # calc offspring fitness
        for o in range(no):

            np.random.seed(0)
            action, empirical, RT = simulate(selector, bs[o+1], ss[o+1], wds[o+1], sample_post, sample_other, prior_as_start, npi=npi)

            post_fits[i,o+1] = np.abs((posts - empirical)/posts).mean(axis=1).mean()
            agr_meds[i,o+1] = np.median(RT[1,:])
            penalties[i,o+1], med_penalties[i,o+1], fit_penalties[i,o+1] = calc_penalty(post_fits[i,o+1], agr_meds[i,o+1])

        # select best candidate
        
        print("\npost_fits", post_fits[i,:])
        # print("fit_penalties", fit_penalties)
        # print("med_penalties", med_penalties)
        # print("penalties", penalties)
        best = np.argmin(penalties[i,:])
        b, bs[best] = [bs[best]]*2
        wd, wds[best] = [wds[best]]*2
        best_wds[i] = wd
        s, ss[best] = [ss[best]]*2
        post_fits[i+1,0] = post_fits[i,best]
        fit_penalties[i+1,0] = fit_penalties[i,best]
        agr_meds[i+1,0] = agr_meds[i,best]
        med_penalties[i+1,0] = med_penalties[i,best]
        penalties[i+1,0] = penalties[i,best]
        best_fit = post_fits[i,best]
        print(i, best, best_fit)
        print(b,wd)

        if last_best_fit == best_fit:
            best_fit_counter += 1
        else:
            best_fit_counter = 0

        last_best_fit = best_fit

        print(best_fit_counter)

        if(best_fit_counter >= 10):
            total = total + best_fit_counter
            factor = factor/5
            best_fit_counter = 0
        i += 1

    results = {
        'fits': post_fits, 
        'penalties': fit_penalties, 
        'agr_meds': agr_meds, 
        'best_wds': best_wds,
        'params': initial_params,
    }

    return results

def sim_plot_rt_post(initial_params, trials = 2000, save_fig = False):

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
    posts = np.zeros([len(modes),npi])    # translate the posteriors
    post = np.asarray(test_vals)[ind,:,0]  # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)

    print( np.abs((posts - empirical)/posts).mean(axis=1).mean())

    if not save_fig:
        plt.show()
    else:
        plt.savefig(make_title(initial_params, add='figure'), dpi=300)

def plot_from_file(ttl):
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = plt.subplot(211)
    ax2 = plt.subplot(212)
    npi, selector, pars, regime, s, wd, b = extract_params_from_ttl(ttl)
    
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

    ind = np.where(polss == npi)[0][0]
    posts = np.zeros([len(modes), 3])    # translate the posteriors
    post = np.asarray(test_vals)[ind,:,0]  # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)

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
s =  0.0009
initial_params = [3, 'ardm', 5, [0.4926206128104312, 0.009], params_dict['like_prior0'] + ['like_prior0']]
initial_params = [3, 'rdm', 3, [330*s, s], params_dict[regime] + [regime]]
# initial_params = [3, 'ardm', 5, [0.4926206128104312, 0.009], params_dict['like_prior0'] + ['like_prior0']]



s =  0.0009
# for b in [1.025, 1.035, 1.045, 1.05]:
for b in [1.01]:
    initial_params = [2, 'ddm', b, [0.09983708592967723, s], params_dict[regime] + [regime]]
    sim_plot_rt_post(initial_params, trials=10000, save_fig=True)

var =  0.0009
for s in(var + var*np.arange(0.005, 0.015, 0.002)):
    initial_params = [2, 'ddm', b, [0.09983708592967723, s], params_dict[regime] + [regime]]
    sim_plot_rt_post(initial_params, trials=10000, save_fig=True)

ratio = 0.09983708592967723 / 0.0009
for wd in(s*(ratio + ratio*np.arange(0.005, 0.015, 0.002))):
    initial_params = [2, 'ddm', b, [0.09983708592967723, s], params_dict[regime] + [regime]]
    sim_plot_rt_post(initial_params, trials=10000, save_fig=True)
# find_best_params(initial_params)

'''serialise find best parameters'''
results = []
for method in methods:
    print(method)
    for key in params_dict:
        print(key)
        initial_params[4] = params_dict[key] + [key]
        results.append(find_best_params(initial_params))


with open('fit_results.txt', 'wb') as fp:
    pickle.dump(results,fp)



# find_best_params(initial_params)
        
# initial_params[2] = 1.5
# initial_params[3][0] = 1.5

'''
SIMULATE AND PLOT RT DIST WITH POST APPROX
'''
    
# sim_plot_rt_post(initial_params)
# print('poop')



'''
EXPLORE AROUND TOP CANDIDATES

'''

# %%
df = load_data()
# print the fit values for all the best fits
df['opt_fit_group'] = df.groupby(['npi', 'selector','regime'])['fit'].transform('min')
df['optimal'] =(df['fit'] == df['opt_fit_group'])*1
best_fits = df[df['optimal'] == 1]
print(best_fits[['selector','regime','b', 's', 'w', 'fit','agr_mode','hab_mode','goal_mode', 'conf_mode']].sort_values(['selector','fit']))


trials = 1000
for ind, row in best_fits.iterrows():
    if row.selector == '0.0':
        pass
    else:
        
        npi, selector, par, reg, so, wdo, bo = extract_params_from_ttl(row.title)
        wd = 1.07811
        b = 5 #1.1
        i = np.where(polss == npi)[0][0]
        # trials = 10000
        ratio = wd/so - 10
        so = so
        wd = so*ratio
        np.random.seed(0)
        sim_plot_rt_post([npi, selector, b, [wd,so], par + [reg]], trials=1000)
        
        # # print('original fit: ', fit)
        # ratio = wdo/so
        # # bs = bo + np.arange(0.10, 0.3, 0.01)
        # bo = 1.1
        # wd = 1.0781099999999995
        # ratios = np.arange(ratio - 2*ratio*0.01, ratio - ratio*0.01, ratio*0.001)
        # wds = ratios*so
        # print(wdo, wds)
        # np.random.seed(0)
        # fits = np.zeros(wds.size+1)


        # actions, empirical,RT, fits[0] = simulate(i,selector, bo, so, wdo, par[0], par[1], par[2], calc_fit=True)
        # for wdi, wd in enumerate(wds):
        #     print('wdi, wd: ', wdi, wd)
        #     np.random.seed(0)
        #     actions, empirical, RT, fits[wdi+1] = simulate(i,selector, bo, so, wd, par[0], par[1], par[2], calc_fit=True)
        #     print(fits)

        # print(row[['selector','regime','s', 'w', 'fit','agr_mode','hab_mode','goal_mode', 'conf_mode']])
        # print(row.fit)
        # plot_from_file(row.title)
        # break






#%%
# posts = np.zeros([len(modes), 3])      # translate the posteriors
# post = np.asarray(test_vals)[0,:,0]    # into a numpy array

# for indx, p in enumerate(post):
#     posts[indx,:] = np.asarray(p)

# for ind, row in df[df['optimal'] == 1].iterrows():
#     npi, selector, pars, regime, s, wd, b = extract_params_from_ttl(row.title)
#     ratio = wd/s
#     # wds = np.arange(-10,10,1)
#     # wds = wds[wds != 0]
#     wds = np.arange(0,3)
#     wds = (wds + ratio)*s
#     sample_post, sample_other, prior_as_start = params_dict[regime]
    
#     for ind, wd in enumerate(wds): 
#         ttl = '_'.join(['npi', str(npi), selector, regime, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 
#         print('\n' + ttl)
#         empirical = np.zeros([nmodes, npi])
#         RT = np.zeros([nmodes, trials])

#         for m, mode in enumerate(modes):
#             i = np.where(polss == npi)[0][0]
#             prior = test_vals[i][m][1]
#             like = test_vals[i][m][2]
#             post = test_vals[i][m][0]
#             # print('variance:', s)
#             actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
#                             prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
#                             var=s, wd=wd, b=b)
#             actions = np.asarray(actions)
#             actions = actions[actions != -1]
#             actions = actions.tolist()
#             empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
#             RT[m,:] = ac_sel.RT.squeeze()
            
#         post_fit = np.abs((posts - empirical)/posts).mean(axis=1).mean()
#         print(post_fit)
#         ttl = '_'.join(['npi', str(npi), selector, regime, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), '.txt']) 

#         with open(path + ttl, 'wb') as fp:

#             dict = {
#                 'RT': RT,
#                 'empirical': empirical,
#                 'parameters': parameter_names,
#                 'parameter_values':p
#             }

#             pickle.dump(dict, fp)




# generate_data()


'''
PLOT APPROXIMATION FROM DATA FILE AND STATS
'''

ttl = 'npi_3_rdm_standard_b_1_wd_0.025_s_2.5e-05_.txt'


