# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import action_selection as asl
import seaborn as sns
import pandas as pd
from scipy.stats import entropy
plt.style.use('seaborn-whitegrid')
from pandas.plotting import table
from misc import params_list, simulate, make_title, test_vals, calc_dkl, cols
import pickle as pickle
import time
import os as os
import itertools as itertools
import string as string
#%%


#%%
path = os.getcwd() + '\\parameter_data\\'
tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")



'''
function definitions
'''    

test_vals = [[],[],[],[]]

# ///////// setup 3 policies

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


#  ///////////// setup 2 policies



npi = 2
flat = [1./npi]*npi

# conflict
l = [0.8,0.2]
conflict_prior = np.ones(2) - l + 0.1
conflict_prior /= conflict_prior.sum()

prior = np.array(conflict_prior)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[3].append([post,prior,like])

# agreement
l = np.array([0.8,0.2])
agree_prior = l + 0.3
agree_prior /= agree_prior.sum()

prior = np.array(agree_prior)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[3].append([post,prior,like])

# goal
l = [0.8,0.2]
prior = np.array(flat)
like = np.array(l)
post = prior*like
post /= post.sum()
test_vals[3].append([post,prior,like])

# habit
prior = np.array([0.8,0.2])
like = np.array(flat)
post = prior*like
post /= post.sum()
test_vals[3].append([post,prior,like])



def calc_dkl(empirical, post):
    # print(p)
    # print(q)
    dkls = np.zeros(4)
        
    for m in range(4):
        p = empirical[m,:]
        q = post[m,:]
        p[p == 0] = 10**(-300)
        q[q == 0] = 10**(-300)
    
        ln = np.log(p/q)
        if np.isnan(p.dot(ln)):
            raise ValueError('is Nan')
    
        dkls[m] = p.dot(ln)
        
    return dkls
def extract_params_from_ttl(ttl):
    names = ['standard', 'post_prior1', 'post_prior0', 'like_prior1', 'like_prior0']

    params_dict = {
        'standard_b': [False, False, True],
        'post_prior1': [True, False, True], 
        'post_prior0': [True, False, False],
        'like_prior1': [False, True, True], 
        'like_prior0': [False, True, False]
    }
    pars = ttl.split('_')

    for indx, par in enumerate(pars):
        if par == 'b':
            b = float(pars[indx+1])
        if par == 's':
            s = float(pars[indx+1])
        if par == 'wd':
            wd = float(pars[indx+1])

    npi = int(pars[1])
    selector = pars[2]
    regime = '_'.join(pars[3:5])
    pars = params_dict[regime]
    # print(pars)
    return npi, selector, pars, regime, s, wd, b


def extract_params(ttl):
    names = ['standard', 'post_prior1', 'post_prior0', 'like_prior1', 'like_prior0']

    params_dict = {
        'standard_b': [False, False, True],
        'post_prior1': [True, False, True], 
        'post_prior0': [True, False, False],
        'like_prior1': [False, True, True], 
        'like_prior0': [False, True, False]
    }
    pars = ttl.split('_')

    for indx, par in enumerate(pars):
        if par == 'b':
            b = float(pars[indx+1])
        if par == 's':
            s = float(pars[indx+1])
        if par == 'wd':
            wd = float(pars[indx+1])

    npi = int(pars[1])
    selector = pars[2]
    regime = '_'.join(pars[3:5])
    pars = params_dict[regime]
    # print(pars)
    return [npi, selector, b, [wd,s], pars + [regime]]

params_dict = {
    'standard_b': [False, False, True],
    'post_prior1': [True, False, True], 
    'post_prior0': [True, False, False],
    'like_prior1': [False, True, True], 
    'like_prior0': [False, True, False]
}

modes = ['conflict', 'agreement', 'goal', 'habit']
nmodes = len(modes)

params_list = [[False, False, True, 'standard'],\
              [True, False, True, 'post_prior1' ],\
              [True, False, False, 'post_prior0'],\
              [False, True, True, 'like_prior1' ],\
              [False, True, False, 'like_prior0']]


# ss = [0.01, 0.03, 0.05, 0.07, 0.1]
# wds = np.arange(0.5,2.5,0.5)
# bs = np.arange(1, 2.5, 0.5)

# ss = [0.01, 0.03, 0.05, 0.07, 0.1]
# wds = [0.7, 0.9, 1.3, 1.7, 1.9, 2.1, 2.3]
# bs = [1, 1.3, 1.7, 1.9, 2.1, 2.3]

import itertools as itertools
import os as os
from scipy import stats

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


pols = np.array([3]) #,8,81]
polss =  np.array([3,8,81])
path = os.getcwd() + '\\parameter_data\\'

par_list = []
methods = ['rdm', 'ardm']
parameter_names = ['npi', 'methods', 'b', 'wd', 's', 'params_list']
for p in itertools.product(pols, methods, bs, drift_var, params_list):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]])



def load_data():
    files = os.listdir(path)

    posts = np.zeros([len(modes), 3])    # translate the posteriors
    post = np.asarray(test_vals)[0,:,0]    # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    n = len(files)
    npis = np.zeros(n,dtype="int32")
    selectors = np.zeros(n).tolist()
    bs = np.zeros(n, dtype="int32")
    wds = np.zeros(n)
    ss = np.zeros(n)
    regimes = np.zeros(n).tolist()
    post_fit = np.zeros(n)
    conf_mode = np.zeros(n)
    agr_mode = np.zeros(n)
    goal_mode = np.zeros(n)
    hab_mode = np.zeros(n)
    conf_mean = np.zeros(n)
    agr_mean = np.zeros(n)
    goal_mean = np.zeros(n)
    hab_mean = np.zeros(n)
    conf_median = np.zeros(n)
    agr_median = np.zeros(n)
    goal_median = np.zeros(n)
    hab_median = np.zeros(n)
    ttls = ['fuck off']
    ttls = np.zeros(n,dtype="object")
    for ind, f in enumerate(files):
        if f != 'old':
            npis[ind] , selectors[ind], [sample_post, sample_other, prior_as_start], regimes[ind], ss[ind], wds[ind], bs[ind] = \
                extract_params_from_ttl(f)

            # print('\n', f)
            # print(extract_params_from_ttl(f))
            
            with open(path + f, 'rb') as fp:
                data = pickle.load(fp)

            empirical = np.asarray(data['empirical'])
            RTs = np.asarray(data['RT'])
            conf_mode[ind], agr_mode[ind], goal_mode[ind], hab_mode[ind] = np.asarray(stats.mode(RTs, axis=1)[0]).ravel()
            conf_mean[ind], agr_mean[ind], goal_mean[ind], hab_mean[ind] = RTs.mean(axis=1)
            conf_median[ind], agr_median[ind], goal_median[ind], hab_median[ind] = np.median(RTs, axis=1)
            ttls[ind] = f
            post_fit[ind] = np.abs((posts - empirical)/posts).mean(axis=1).mean()

    data = {'npi': npis,
                 'selector':selectors,
                 'b': bs,
                 'w': wds,
                 's': ss,
                 'regime': regimes,
                 'fit': post_fit,
                 'conf_mode':conf_mode,
                 'agr_mode': agr_mode,
                 'goal_mode': goal_mode,
                 'hab_mode': hab_mode,
                 'conf_mean': conf_mean,
                 'agr_mean': agr_mean,
                 'goal_mean': goal_mean,
                 'hab_mean': hab_mean,
                 'conf_median': conf_median,
                 'agr_median': agr_median,
                 'goal_median': goal_median,
                 'hab_median': hab_median,
                 'title': ttls
            }
    df = pd.DataFrame(data)
    # return best_fit, diff_best
    return df



def load_data_from_ttl():

    posts = np.zeros([len(modes), 3])    # translate the posteriors
    post = np.asarray(test_vals)[0,:,0]    # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    n = len(par_list)
    npis = np.zeros(n,dtype="int32")
    selectors = np.zeros(n).tolist()
    bs = np.zeros(n, dtype="int32")
    wds = np.zeros(n)
    ss = np.zeros(n)
    regimes = np.zeros(n).tolist()
    post_fit = np.zeros(n)
    conf_mode = np.zeros(n)
    agr_mode = np.zeros(n)
    goal_mode = np.zeros(n)
    hab_mode = np.zeros(n)
    conf_mean = np.zeros(n)
    agr_mean = np.zeros(n)
    goal_mean = np.zeros(n)
    hab_mean = np.zeros(n)
    conf_median = np.zeros(n)
    agr_median = np.zeros(n)
    goal_median = np.zeros(n)
    hab_median = np.zeros(n)
    ttls = []

    for ind, p in enumerate(par_list):
        # print(ind)

        npis[ind] = p[0]
        selectors[ind] = p[1]
        bs[ind] = p[2]
        wds[ind] = p[3]
        ss[ind] = p[4]
        sample_post, sample_other, prior_as_start, regimes[ind] = p[5]


        ttl = '_'.join(['npi', str(npis[ind]), selectors[ind], regimes[ind] , 'b' ,str(bs[ind]), 'wd',\
                        str(wds[ind]), 's', str(ss[ind]), '.txt']) 
        
        with open(path + ttl, 'rb') as fp:
            data = pickle.load(fp)

        empirical = np.asarray(data['empirical'])
        RTs = np.asarray(data['RT'])
        conf_mode[ind], agr_mode[ind], goal_mode[ind], hab_mode[ind] = np.asarray(stats.mode(RTs, axis=1)[0]).ravel()
        conf_mean[ind], agr_mean[ind], goal_mean[ind], hab_mean[ind] = RTs.mean(axis=1)
        conf_median[ind], agr_median[ind], goal_median[ind], hab_median[ind] = np.median(RTs, axis=1)
        ttls.append(ttl)
        post_fit[ind] = np.abs((posts - empirical)/posts).mean(axis=1).mean()

    data = {'npi': npis,
                 'selector':selectors,
                 'b': bs,
                 'w': wds,
                 's': ss,
                 'regime': regimes,
                 'fit': post_fit,
                 'conf_mode':conf_mode,
                 'agr_mode': agr_mode,
                 'goal_mode': goal_mode,
                 'hab_mode': hab_mode,
                 'conf_mean': conf_mean,
                 'agr_mean': agr_mean,
                 'goal_mean': goal_mean,
                 'hab_mean': hab_mean,
                 'conf_median': conf_median,
                 'agr_median': agr_median,
                 'goal_median': goal_median,
                 'hab_median': hab_median,
                 'title': ttls
            }
    df = pd.DataFrame(data)
    # return best_fit, diff_best
    return df

# def load_data():

#     posts = np.zeros([len(modes), 3])    # translate the posteriors
#     post = np.asarray(test_vals)[0,:,0]    # into a numpy array

#     for indx, p in enumerate(post):
#         posts[indx,:] = np.asarray(p)
    
#     n = 3502
#     n = len(par_list)
#     npis = np.zeros(n,dtype="int32")
#     selectors = np.zeros(n).tolist()
#     bs = np.zeros(n, dtype='int32')
#     wds = np.zeros(n)
#     ss = np.zeros(n)
#     regimes = np.zeros(n).tolist()
#     post_fit = np.zeros(n)
#     conf_mode = np.zeros(n)
#     agr_mode = np.zeros(n)
#     goal_mode = np.zeros(n)
#     hab_mode = np.zeros(n)
#     conf_mean = np.zeros(n)
#     agr_mean = np.zeros(n)
#     goal_mean = np.zeros(n)
#     hab_mean = np.zeros(n)
#     conf_median = np.zeros(n)
#     agr_median = np.zeros(n)
#     goal_median = np.zeros(n)
#     hab_median = np.zeros(n)
#     ttls = []

#     for ind, p in enumerate(par_list):
#         if ind < 3502:
#             npis[ind] = p[0]
#             selectors[ind] = p[1]
#             bs[ind] = p[2]
#             wds[ind] = p[3][0]
#             ss[ind] = p[3][1]
#             sample_post, sample_other, prior_as_start, regimes[ind] = p[4]


#             ttl = '_'.join(['npi', str(npis[ind]), selectors[ind], regimes[ind] , 'b' ,str(bs[ind]), 'wd',\
#                             str(wds[ind]), 's', str(ss[ind]), '.txt']) 
            
#             with open(path + ttl, 'rb') as fp:
#                 data = pickle.load(fp)

#             empirical = np.asarray(data['empirical'])
#             RTs = np.asarray(data['RT'])
#             conf_mode[ind], agr_mode[ind], goal_mode[ind], hab_mode[ind] = np.asarray(stats.mode(RTs, axis=1)[0]).ravel()
#             conf_mean[ind], agr_mean[ind], goal_mean[ind], hab_mean[ind] = RTs.mean(axis=1)
#             conf_median[ind], agr_median[ind], goal_median[ind], hab_median[ind] = np.median(RTs, axis=1)
#             ttls.append(ttl)
#             post_fit[ind] = np.abs((posts - empirical)/posts).mean(axis=1).mean()

#     data = {'npi': npis,
#                 'selector':selectors,
#                 'b': bs,
#                 'w': wds,
#                 's': ss,
#                 'regime': regimes,
#                 'fit': post_fit,
#                 'conf_mode':conf_mode,
#                 'agr_mode': agr_mode,
#                 'goal_mode': goal_mode,
#                 'hab_mode': hab_mode,
#                 'conf_mean': conf_mean,
#                 'agr_mean': agr_mean,
#                 'goal_mean': goal_mean,
#                 'hab_mean': hab_mean,
#                 'conf_median': conf_median,
#                 'agr_median': agr_median,
#                 'goal_median': goal_median,
#                 'hab_median': hab_median,
#                 'title': ttls
#             }
#     df = pd.DataFrame(data)
#         # return best_fit, diff_best
#     return df


cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
polss = np.asarray([3,8,81,2])


def simulate(selector, b, s, wd, A, sample_post, sample_other, prior_as_start, plot=False, calc_fit=False,npi=3, trials=1000):
    empirical = np.zeros([nmodes, npi])
    RT = np.zeros([nmodes, trials])

    if plot:
        x_positions = []
        for i in range(4):
            x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

        fig, ax = plt.subplots(2,1)

    for m, mode in enumerate(modes):
        i = np.where(polss == npi)[0][0]
        prior = test_vals[i][m][1]
        like = test_vals[i][m][2]
        post = test_vals[i][m][0]
        # print('variance:', s)
        actions, ac_sel = run_action_selection(selector, prior, like, post, trials,\
                        prior_as_start=prior_as_start, sample_other=sample_other, sample_post=sample_post,\
                        var=s, wd=wd, b=b, A=A)

        actions = np.asarray(actions)
        actions = actions[actions != -1]
        actions = actions.tolist()
        empirical[m,:] = (np.bincount(actions + [x for x in range(npi)]) - 1) / len(actions)
        RT[m,:] = ac_sel.RT.squeeze()

        if plot:
            print('dont do this')
            x_pos = x_positions[m]
            lab =' '.join([mode, 'mode',  str(stats.mode(ac_sel.RT)[0][0][0]), 'median', str(np.median(ac_sel.RT)), 'mean', str(ac_sel.RT.mean())])
            ax[0].hist(ac_sel.RT, bins=100, alpha=0.5, label=lab)
            
            if m == 0:
                ax[1].bar(x_pos, post, alpha=0.5, color='k', label = "post" )
            else:
                ax[1].bar(x_pos, post, alpha=0.5, color='k')

            ax[1].bar(x_pos, empirical[m,:], label=mode + ' empir', alpha=0.5, color=cols[m])

    return RT, empirical

    

def make_title(params,add_text=None, extra_param = None, format='.png'):
    npi = params[0]
    selector = params[1]
    b = params[2]
    # wd = params[3][0]
    # s = params[3][1]
    wd= params[3]
    s = params[4]
    a = params[5]
    sample_post, sample_other, prior_as_start, reg = params[6]

    if add_text == None:
        if extra_param == None:
            ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd',\
                    str(wd), 's', str(s), format])
        else:

            ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd',\
                    str(wd), 's', str(s), extra_param[0], str(extra_param[1]), format])
    else:
        if extra_param == None:
            ttl = '_'.join([add_text, 'npi', str(npi), selector, reg, 'b' ,str(b), 'wd',\
                    str(wd), 's', str(s), format])
        else: 

            ttl = '_'.join([add_text, 'npi', str(npi), selector, reg, 'b' ,str(b), 'wd',\
                    str(wd), 's', str(s), extra_param[0], str(extra_param[1]), format])
   
    return ttl


def load_file(ttl):
    with open (ttl, 'rb') as fp:
        data = pickle.load(fp)
    
    return data


def test(df,size=4000):
    if not df.shape[0] == size:
        raise ValueError('WRONG SELECTION')

def load_file(ttl):
    with open (ttl, 'rb') as fp:
        data = pickle.load(fp)
    
    return data

def extract_params(ttl):
    names = ['standard', 'post_prior1', 'post_prior0', 'like_prior1', 'like_prior0']

    params_dict = {
        'standard_b': [False, False, True],
        'post_prior1': [True, False, True], 
        'post_prior0': [True, False, False],
        'like_prior1': [False, True, True], 
        'like_prior0': [False, True, False]
    }
    pars = ttl.split('_')
    a_present = False
    for indx, par in enumerate(pars):
        if par == 'b':
            if not len(pars[indx+1]) == 1: 
                b = float(pars[indx+1])
            else:
                b = int(pars[indx+1])
        if par == 's':
            s = float(pars[indx+1])
        if par == 'wd':
            if not len(pars[indx+1]) == 1: 
                wd = float(pars[indx+1])
            else:
                wd = int(pars[indx+1])
        if par == 'a':
            a_present = True
            if not len(pars[indx+1]) == 1: 
                a = float(pars[indx+1])
            else:
                a = int(pars[indx+1])
            # a = float(pars[indx+1])
    # print(pars)            
    npi = int(pars[1])
    selector = pars[2]
    regime = '_'.join(pars[3:5])
    pars = params_dict[regime]
    
    if regime == 'standard_b':
        regime = 'standard'
    if a_present:
        return [npi, selector, b, wd,s, a, pars + [regime]]
    else:
        return [npi, selector, b, wd,s, 1, pars + [regime]]
                
def extract_post(npi=3, nmodes=4):
    x_positions = []
    for i in range(nmodes):
        x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])
        
    polss = np.asarray([3,8,81,2])
    i = np.where(polss == npi)[0][0]
    

    posts = np.zeros([nmodes, npi])    # translate the posteriors
    post = np.asarray(test_vals)[i,:,0]    # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    return posts, x_positions

def load_fits(trials=1000):

    tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")
    names = ['npi', 'selector','b','w','s','A', 'regime', 'post_fit','individual_fit','avg_fit','ID','file_ttl','stats']
    nmodes = 4
    path = os.getcwd() + '/parameter_data/'
    files = os.listdir(path)
    total = len(files)
    npis = np.zeros(total)
    selectors = np.zeros(total, dtype='object')
    bs = np.zeros(total, dtype="float")
    ws = np.zeros(total)
    ss = np.zeros(total)
    As = np.zeros(total)
    regimes = np.zeros(total, dtype="object")
    post_fit = np.zeros(total)
    individual_fit = np.zeros(total, dtype="object")
    post_fit_avg = np.zeros(total)
    polss = np.array([3,8,81])
    ID = np.zeros(total)
    titles = np.zeros(total, dtype="object")
    stats = np.zeros(total, dtype="object")
    fucked = []
    for ind, f in enumerate(files):
        if ind < total:
            p = extract_params(f)
            # print(ind)
            # print(f)
            npi = p[0]
            y = np.where(polss == npi)[0][0]    
            posts = np.zeros([nmodes, npi])    # translate the posteriors
            post = np.asarray(test_vals)[y,:,0]    # into a numpy array
            
            for indx, ps in enumerate(post):
                posts[indx,:] = np.asarray(ps)
                
            selector = p[1]
            b = p[2]
            w = p[3]
            s = p[4]
            A = p[5]
            post, other, prior, regime = p[6]
            regime = regime
            # ttl =  make_title(p, extra_param = ['a',str(A)], format='.txt')
            # print(ind)
            with open(path + f, 'rb')as fp:
                data = pickle.load(fp)
            
            RT = data['RT']
            
            
            rt_df = {
                'rts': RT.ravel() ,
                'mode': tests.repeat(trials)
                }
            
            rt_df = pd.DataFrame(rt_df)
            
            stats[ind] = rt_df.groupby(['mode']).agg(
                mean = ("rts","mean"),
                median = ("rts", "median"),
                var = ("rts", "var"),
                skew =("rts", "skew"),
            )
            
            empirical = data['empirical']
            
            if np.isnan(np.unique(data['empirical'])[-1]):
                fucked.append(f)
                fits = np.nan
                post_fit[ind] =  np.nan
                post_fit_avg[ind] =  np.nan
                individual_fit[ind] = np.nan
            else:
                fits = calc_dkl(empirical, posts)
                post_fit[ind] =  fits.mean()
                post_fit_avg[ind] =  np.abs((posts - empirical)/posts).mean(axis=1).mean()
                individual_fit[ind] = fits
            
            ID[ind] = ind
            npis[ind] = npi
            selectors[ind] = selector
            bs[ind] = b
            ws[ind] = w
            ss[ind] = s
            As[ind] = A
            regimes[ind] = regime
            bs[ind] = b
            titles[ind] = f
        
    cols = [npis, selectors, bs, ws, ss, As, regimes, post_fit, individual_fit, post_fit_avg,ID,titles,stats]
    dict_data = {}
    
    for c, col in enumerate(cols):
        dict_data[names[c]] = col
    
    df = pd.DataFrame(dict_data)
    return df,fucked

def load_fits_from_data(pars, trials=1000):

    tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")
    names = ['npi', 'selector','b','w','s','A', 'regime', 'post_fit','individual_fit','avg_fit','ID','file_ttl','stats']
    nmodes = 4
    path = os.getcwd() + '\\parameter_data\\'
    files = os.listdir(path)
    total = len(pars)
    npis = np.zeros(total)
    selectors = np.zeros(total, dtype='object')
    bs = np.zeros(total, dtype="float")
    ws = np.zeros(total)
    ss = np.zeros(total)
    As = np.zeros(total)
    regimes = np.zeros(total, dtype="object")
    post_fit = np.zeros(total)
    individual_fit = np.zeros(total, dtype="object")
    post_fit_avg = np.zeros(total)
    polss = np.array([3,8,81])
    ID = np.zeros(total)
    titles = np.zeros(total, dtype="object")
    stats = np.zeros(total, dtype="object")
    fucked = []
    
    for ind, p in enumerate(pars):
        if ind < total:
            # print(ind)
            # print(f)
            npi = p[0]
            print(p)
            print(npi)
            print(polss)
            print(np.where(polss == npi))
            y = np.where(polss == npi)[0][0]    
            posts = np.zeros([nmodes, npi])    # translate the posteriors
            post = np.asarray(test_vals)[y,:,0]    # into a numpy array
            
            for indx, ps in enumerate(post):
                posts[indx,:] = np.asarray(ps)
                
            selector = p[1]
            b = p[2]
            w = p[3]
            s = p[4]
            A = p[5]
            post, other, prior, regime = p[6]
            regime = regime
            ttl =  make_title(p, extra_param = ['a',str(A)], format='.txt')
            # print(ind)
            with open(path + ttl, 'rb')as fp:
                data = pickle.load(fp)
            
            RT = data['RT']
            
            
            rt_df = {
                'rts': RT.ravel() ,
                'mode': tests.repeat(trials)
                }
            
            rt_df = pd.DataFrame(rt_df)
            
            stats[ind] = rt_df.groupby(['mode']).agg(
                mean = ("rts","mean"),
                median = ("rts", "median"),
                var = ("rts", "var"),
                skew =("rts", "skew"),
            )
            
            empirical = data['empirical']
            
            if np.isnan(np.unique(data['empirical'])[-1]):
                fucked.append(f)
                fits = np.nan
                post_fit[ind] =  np.nan
                post_fit_avg[ind] =  np.nan
                individual_fit[ind] = np.nan
            else:
                fits = calc_dkl(empirical, posts)
                post_fit[ind] =  fits.mean()
                post_fit_avg[ind] =  np.abs((posts - empirical)/posts).mean(axis=1).mean()
                individual_fit[ind] = fits
            
            ID[ind] = ind
            npis[ind] = npi
            selectors[ind] = selector
            bs[ind] = b
            ws[ind] = w
            ss[ind] = s
            As[ind] = A
            regimes[ind] = regime
            bs[ind] = b
            titles[ind] = ttl
        
    cols = [npis, selectors, bs, ws, ss, As, regimes, post_fit, individual_fit, post_fit_avg,ID,titles,stats]
    dict_data = {}
    
    for c, col in enumerate(cols):
        dict_data[names[c]] = col
    
    df = pd.DataFrame(dict_data)
    return df,fucked


def load_data(trials=1000):
    nmodes = 4
    path = os.getcwd() + '/parameter_data/'
    files = os.listdir(path)
    total = len(files)*4000
    npis = np.zeros(total)
    selectors = np.zeros(total, dtype='object')
    bs = np.zeros(total, dtype="float")
    ws = np.zeros(total)
    ss = np.zeros(total)
    As = np.zeros(total)
    regimes = np.zeros(total, dtype="object")
    rts = np.zeros(total)
    modes = np.zeros(total,dtype="object")
    post_fit = np.zeros(total)
    individual_fit = np.zeros(total)
    tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")
    names = ['npi', 'selector','b','w','s','A', 'regime', 'rts', 'mode', 'post_fit','individual_fit','avg_fit','ID','file_ttl']
    post_fit_avg = np.zeros(total)
    polss = np.array([3,8,81])
    ID = np.zeros(total)
    titles = np.zeros(total, dtype="object")
    
    for ind, f in enumerate(files):
        if ind == 1169:
            p = extract_params(f)
            # print(ind)
            # print(f)
            npi = np.array([p[0]])
    
            y = np.where(polss == npi)[0][0]    
            posts = np.zeros([nmodes, npi[0]])    # translate the posteriors
            post = np.asarray(test_vals)[y,:,0]    # into a numpy array
        
            for indx, ps in enumerate(post):
                posts[indx,:] = np.asarray(ps)
                
            selector = np.asarray(p[1], dtype="object")
            b = np.array(p[2], dtype="int32")
            w  = np.array(p[3])
            s  = np.array(p[4])
            A  = np.array(p[5])
            post, other, prior, regime = p[6]
            regime = np.asarray(regime)
            # ttl =  make_title(p, extra_param = ['a',str(A)], format='.txt')
    
            with open(path + f, 'rb')as fp:
                data = pickle.load(fp)
    
            RT = data['RT']
            empirical = data['empirical']
            print(ind, f)
            if np.isnan(np.unique(data['empirical'])[0]):
                fits = np.array(np.nan)
                avg_fit = np.array(np.nan)
                post_fit[ind*trials*nmodes: (ind+1)*trials*nmodes] =  fits.repeat(trials*nmodes)
                post_fit_avg[ind*trials*nmodes: (ind+1)*trials*nmodes] =  avg_fit.repeat(trials*nmodes)
                individual_fit[ind*trials*nmodes: (ind+1)*trials*nmodes] = fits.repeat(trials*nmodes)
            else:
                fits = calc_dkl(empirical, posts)
                avg_fit = np.abs((posts - empirical)/posts).mean(axis=1).mean()
                post_fit[ind*trials*nmodes: (ind+1)*trials*nmodes] =  fits.mean().repeat(trials*nmodes)
                post_fit_avg[ind*trials*nmodes: (ind+1)*trials*nmodes] =  avg_fit.repeat(trials*nmodes)
                individual_fit[ind*trials*nmodes: (ind+1)*trials*nmodes] = fits.repeat(trials)
            ID[ind*trials*nmodes: (ind+1)*trials*nmodes] = np.array([ind]).repeat(trials*nmodes)
            npis[ind*trials*nmodes: (ind+1)*trials*nmodes] = npi.repeat(trials*nmodes)
            selectors[ind*trials*nmodes: (ind+1)*trials*nmodes] = selector.repeat(trials*nmodes)
            bs[ind*trials*nmodes: (ind+1)*trials*nmodes] = b.repeat(trials*nmodes)
            ws[ind*trials*nmodes: (ind+1)*trials*nmodes] = w.repeat(trials*nmodes)
            ss[ind*trials*nmodes: (ind+1)*trials*nmodes] = s.repeat(trials*nmodes)
            As[ind*trials*nmodes: (ind+1)*trials*nmodes] = A.repeat(trials*nmodes)
            regimes[ind*trials*nmodes: (ind+1)*trials*nmodes] = regime.repeat(trials*nmodes)
            bs[ind*trials*nmodes: (ind+1)*trials*nmodes] = b.repeat(trials*nmodes)
            modes[ind*trials*nmodes: (ind+1)*trials*nmodes] = tests.repeat(trials)
            rts[ind*trials*nmodes: (ind+1)*trials*nmodes] = RT.ravel()
            titles[ind*trials*nmodes: (ind+1)*trials*nmodes] = np.array([f],dtype="object").repeat(nmodes*trials)
    
    cols = [npis, selectors, bs, ws, ss, As, regimes, rts, modes, post_fit, individual_fit, post_fit_avg,ID,titles]
    dict_data = {}

    for c, col in enumerate(cols):
        dict_data[names[c]] = col

    df = pd.DataFrame(dict_data)
    return df


def load_data_from_ttl(par_list, trials=1000):
    nmodes = 4
    path = os.getcwd() + '/parameter_data/'
    files = os.listdir(path)
    total = len(par_list)*nmodes*trials
    npis = np.zeros(total)
    selectors = np.zeros(total, dtype='object')
    bs = np.zeros(total)
    ws = np.zeros(total)
    ss = np.zeros(total)
    As = np.zeros(total)
    regimes = np.zeros(total, dtype="object")
    rts = np.zeros(total)
    modes = np.zeros(total,dtype="object")
    post_fit = np.zeros(total)
    individual_fit = np.zeros(total)
    tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")
    names = ['npi', 'selector','b','w','s','A', 'regime', 'rts', 'mode', 'post_fit','individual_fit','avg_fit','ID','file_ttl']
    post_fit_avg = np.zeros(total)
    polss = np.array([3,8,81])
    ID = np.zeros(total)
    titles = np.zeros(total, dtype="object")

    for ind, p in enumerate(par_list):
        npi = np.array([p[0]])
        selector = np.asarray(p[1], dtype="object")
        b = np.array(p[2], dtype="int32")
        w  = np.array(p[3])
        s  = np.array(p[4])
        A  = np.array(p[5])
        post, other, prior, regime = p[6]
        regime = np.asarray(regime)
        
        # ttl = '_'.join(['npi', str(npi), selector, regime, 'b' ,str(b), 'wd' ,str(w), 's', str(s), '.txt']) 
        path = os.getcwd() + '/parameter_data/'
        ttl =  make_title(p, format='.txt')

        y = np.where(polss == npi)[0][0]    
        posts = np.zeros([nmodes, npi[0]])    # translate the posteriors
        post = np.asarray(test_vals)[y,:,0]    # into a numpy array
    
        for indx, ps in enumerate(post):
            posts[indx,:] = np.asarray(ps)
            

        with open(path + ttl, 'rb')as fp:
            data = pickle.load(fp)

        RT = data['RT']
        empirical = data['empirical']
        fits = calc_dkl(empirical, posts)
        avg_fit = np.abs((posts - empirical)/posts).mean(axis=1).mean()
        ID[ind*trials*nmodes: (ind+1)*trials*nmodes] = np.array([ind]).repeat(trials*nmodes)
        post_fit[ind*trials*nmodes: (ind+1)*trials*nmodes] =  fits.mean().repeat(trials*nmodes)
        post_fit_avg[ind*trials*nmodes: (ind+1)*trials*nmodes] =  avg_fit.repeat(trials*nmodes)
        individual_fit[ind*trials*nmodes: (ind+1)*trials*nmodes] = fits.repeat(trials)
        npis[ind*trials*nmodes: (ind+1)*trials*nmodes] = npi.repeat(trials*nmodes)
        selectors[ind*trials*nmodes: (ind+1)*trials*nmodes] = selector.repeat(trials*nmodes)
        bs[ind*trials*nmodes: (ind+1)*trials*nmodes] = b.repeat(trials*nmodes)
        ws[ind*trials*nmodes: (ind+1)*trials*nmodes] = w.repeat(trials*nmodes)
        ss[ind*trials*nmodes: (ind+1)*trials*nmodes] = s.repeat(trials*nmodes)
        As[ind*trials*nmodes: (ind+1)*trials*nmodes] = A.repeat(trials*nmodes)
        regimes[ind*trials*nmodes: (ind+1)*trials*nmodes] = regime.repeat(trials*nmodes)
        bs[ind*trials*nmodes: (ind+1)*trials*nmodes] = b.repeat(trials*nmodes)
        modes[ind*trials*nmodes: (ind+1)*trials*nmodes] = tests.repeat(trials)
        rts[ind*trials*nmodes: (ind+1)*trials*nmodes] = RT.ravel()
        titles[ind*trials*nmodes: (ind+1)*trials*nmodes] = np.array([ttl],dtype="object").repeat(nmodes*trials)
    
    cols = [npis, selectors, bs, ws, ss, As, regimes, rts, modes, post_fit, individual_fit, post_fit_avg,ID,titles]
    dict_data = {}

    for c, col in enumerate(cols):
        dict_data[names[c]] = col

    df = pd.DataFrame(dict_data)
    return df


def plot_table(df):
    grouped = df.groupby(['regime', 'mode']).agg(
        mean = ("rts","mean"),
        median = ("rts", "median"),
        var = ("rts", "var"),
        skew =("rts", "skew")
    )
    print(grouped.sort_values(by=["regime","mean"]).round(2))
    return grouped.sort_values(by=["regime","mean"]).round(2)

def plot_figure(df,q):    
    plt.figure()
    sns.histplot(data=df, x='rts', hue="mode", bins=100, alpha=0.5)
    plt.title(q)

def generate_data(par_list,trials = 1000):
    path = os.getcwd() + '/parameter_data/'
    parameter_names = ['npi','selelctor', 'b','w','s','a','regime']
    for ind, p in enumerate(par_list):
        polss = np.asarray([3,8,81,2])
        start = time.perf_counter()
        print(ind)
        npi = p[0]
        selector = p[1]
        b = p[2]
        wd = p[3]
        s = p[4]
        A = p[5]
        sample_post, sample_other, prior_as_start, reg = p[6]
        posts, none = extract_post(npi=npi)

        ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), 'a', str(A), '.txt']) 
        print('\n' + ttl)
        
        RT, empirical = simulate(selector, b,s,wd, A,sample_post, sample_other, prior_as_start, npi=npi,trials=trials)

        print('fit', np.abs((posts - empirical)/posts).mean(axis=1).mean())

        with open(path + ttl, 'wb') as fp:

            dict = {
                'RT': RT,
                'empirical': empirical,
                'parameters': parameter_names,
                'parameter_values':p
            }

            pickle.dump(dict, fp)
        
        stop = time.perf_counter()
        print((stop - start)/60)  

def show_rts(pars,trials=1000):
    df, nanend = load_fits_from_data(pars,trials=trials)

    for ind, row in df.iterrows():

        x_positions = []
        npi=int(row['npi'])
        for i in range(4):
            x_positions.append([x for x in range(i*npi + i, i*npi + i + npi)])

 
        data = load_file(os.getcwd() + '/parameter_data/' + row['file_ttl'])
        print(row['file_ttl'])
        print(row['stats'].sort_values(by=['mode']))

        rt_df = {
            'rts': data['RT'].ravel() ,
            'mode': tests.repeat(trials)
            }
        
        rt_df = pd.DataFrame(rt_df)

        fig, ax = plt.subplots(2,1)

        posts = np.zeros([4, 3])    # translate the posteriors
        post = np.asarray(test_vals)[0,:,0]    # into a numpy array

        for indx, p in enumerate(post):
            posts[indx,:] = np.asarray(p)
    
        sns.histplot(ax=ax[0], data=rt_df, x='rts', hue='mode',bins=150)

        for m in range(4):
            post = posts[m]
            x_pos = x_positions[m]
            ax[1].bar(x_pos, post, alpha=0.5, color='k')

            ax[1].bar(x_pos, data['empirical'][m,:], alpha=0.5, color=cols[m])

        ax[0].set_title(row['regime'])
        plt.show()
        plt.savefig('posterior_approx_' + row['regime'] + '.png', dpi=300)



def save_data(file_name, objects):

    with open(file_name, 'wb') as output_file:
        pickle.dump(objects, output_file)

def load_data(file_name):
    
    with open(file_name, 'rb') as file:
        objects = pickle.load(file)

    return objects

    
def extract_object(obj):

    keys = []
    obj_dict = obj.__dict__

    for key in obj_dict:
        keys.append(key)


    return keys, obj_dict



def run_action_selection(selector, prior, like, post, trials=10, T=2, prior_as_start=True, sample_post=False,\
                         sample_other=False, var=0.01, wd=1, b=1,A=1):
    
    na = prior.shape[0]
    controls = np.arange(0, na, 1)


    if selector == 'rdm':
        # not really over_actions, simply avoids passing controls
        ac_sel = asl.RacingDiffusionSelector(trials, T, number_of_actions=na, s=var, over_actions=False)
    elif selector == 'ardm':
        # not really over_actions, simply avoids passing controls
        ac_sel = asl.AdvantageRacingDiffusionSelector(trials, T, number_of_actions=na, over_actions=False)
    elif selector == 'nardm':
        # not really over_actions, simply avoids passing controls
        ac_sel = asl.NewAdvantageRacingDiffusionSelector(trials, T, number_of_actions=na, over_actions=False)
    elif selector == 'ddm':
        ac_sel =asl.DDM_RandomWalker(trials, 2,)
        ac_sel
    else: 
        raise ValueError('Wrong or no action selection method passed')
    
    ac_sel.prior_as_starting_point = prior_as_start
    ac_sel.sample_other = sample_other
    ac_sel.sample_posterior = sample_post
    ac_sel.wd = wd
    ac_sel.A = A
    # print(ac_sel.type, ac_sel.A)
    if not selector == 'ddm':
        ac_sel.b = b
    else:
        ac_sel.au = b
        ac_sel.al = -b

    ac_sel.s = np.sqrt(var)
    # print('sdv:', ac_sel.s)
    # print('prior as start, sample_other, sample_post')
    # print(ac_sel.prior_as_starting_point, ac_sel.sample_other, ac_sel.sample_posterior)
    actions = []
    # print(ac_sel.type, ac_sel.sample_posterior, ac_sel.sample_other, ac_sel.prior_as_starting_point, ac_sel.b, ac_sel.wd, ac_sel.s)
    
    for trial in range(trials):
        actions.append(ac_sel.select_desired_action(trial, 0, post, controls, like, prior))   #trial, t, post, control, like, prior
    # print(ac_sel.RT[:20].T)

    return actions, ac_sel

def plot_rts_and_entropy(worlds, hsim, trials=5, na=4, g1=14,g2=10):
    nagents = len(worlds)
    rt = np.zeros([nagents*trials, na])
    agent = np.arange(nagents).repeat(trials)
    trial = np.tile(np.arange(0,trials),nagents)
    h = np.zeros(trials*nagents)
    prior_entropy = np.zeros(trials*nagents) 
    Q_entropy = np.zeros(trials*nagents) 

    accuracy = np.zeros(trials*nagents)
    acc = np.zeros(trials)
    goal_reached = np.zeros(trials)
    for ind, ww in enumerate(worlds):

        print(np.min(ww.agent.perception.dirichlet_pol_params))
        h[ind*trials:(ind+1)*trials] = \
                np.min(ww.agent.perception.dirichlet_pol_params).repeat(trials)
        
        trial_index = np.arange(0,na*trials,4)
        Q = np.asarray(ww.agent.action_selection.drifts)[trial_index]
        prior = np.asarray(ww.agent.action_selection.priors)[trial_index]
        rt[ind*trials:(ind+1)*trials,:] = ww.agent.action_selection.RT
        # print(ww.agent.action_selection.RT.shape)
        Q_entropy[ind*trials:(ind+1)*trials] = entropy(Q, base=2, axis=1)
        prior_entropy[ind*trials:(ind+1)*trials] = entropy(prior, base=2, axis=1)
        goal_reached[:trials//2] = ww.environment.hidden_states[:trials//2,-1]==g1
        goal_reached[trials//2:] = ww.environment.hidden_states[trials//2:,-1]==g2
        print(ww.environment.hidden_states[:,-1])
        acc[:trials//2] = np.cumsum(goal_reached[:trials//2])
        acc[trials//2:] = np.cumsum(goal_reached[trials//2:])
        acc = acc / np.append(np.arange(2,102),np.arange(2,102))
        accuracy[ind*trials:(ind+1)*trials] = acc

    df = pd.DataFrame(rt, columns =['a1','a2','a3','a4'])
    df['trials'] = trial
    df['agent'] = agent
    df['h'] = h
    df['Q_entropy'] = Q_entropy
    df['prior_entropy'] = prior_entropy
    df['accuracy'] = accuracy

    f = plt.figure(figsize=(16,3))
    ax1 = f.add_subplot(1,4,1)
    sns.lineplot(data=df, x="trials",y="a1", hue="h", palette="Dark2", linewidth = 1, marker='o')
    ax1 = f.add_subplot(1,4,2)
    sns.lineplot(data=df, x="trials",y="accuracy", hue="h", palette="Dark2")
    ax1 = f.add_subplot(1,4,3)
    sns.lineplot(data=df, x="trials",y="Q_entropy", hue="h", palette="Dark2")
    ax1 = f.add_subplot(1,4,4)
    sns.lineplot(data=df, x="trials",y="prior_entropy", hue="h", palette="Dark2")



    plt.savefig(os.getcwd() + '\\agent_sims\\good_aprox\\' + hsim + '.png', dpi=300)
    # plt.show()
    plt.close()


# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.animation
# import numpy as np


# fig = plt.figure()
# ax =plt.axes(xlim=(0,81),ylim =(0,1.2))
# line, = ax.plot([],[], lw=2)

# def init_animation():
#     global line
#     line, = ax.plot(x, np.zeros_like(x))
#     ax.set_xlim(0, 81)
#     ax.set_ylim(0,1)


# def animate(i):
#     line.set_ydata(posterior_policies[i,:])
#     fig.suptitle(str(i))
#     return line,

# fig = plt.figure()
# ax = fig.add_subplot(111)
# x = np.arange(0,81)
# # 
# # def init():
# #     line.set_data([],[])
# #     return line, 

# # def animate(i):
# #     x = np.arange(0,81)
# #     y = posterior_policies[i,:]
# #     line.set_data(x,y)

# #     return line,

# ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=200)
# ani.save(r'C:/Users/admin/Desktop/animation.gif', writer='imagemagick', fps=15)


# # anim = animation.FuncAnimation(fig, animate, init_func=init,
# #                                frames=200, interval=20, blit=True)

# # FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])

# # anim.save('basic_animation.mp4', writer=FFwriter)
# # HTML(anim.to_html5_video())