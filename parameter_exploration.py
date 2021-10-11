_# To add a new cell, type '# %%'
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
path = os.getcwd() + '\\parameter_data\\'
tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")

# %%
'''
used parameters
'''
pols = [8,81]
bs = [3]
As = [1]
ws = [1]
ss = [0.0004]
selectors = ['rdm', 'ardm']  #
path = os.getcwd() + '\\parameter_data\\'
tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")

par_list = []

for p in itertools.product(pols, selectors, bs, ws, ss, As, params_list):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]] + [p[5]] + [p[6]])


#%%    
'''
function definitions
'''    


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
                b = float(pars[indx+1])
        if par == 's':
            s = float(pars[indx+1])
        if par == 'wd':
            wd = float(pars[indx+1])
        if par == 'a':
            a_present = True
            a = int(pars[indx+1])
            
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
    path = os.getcwd() + '/parameter_data/'
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
    
    


def generate_data(pars):
    posts, none = extract_post()
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

        ttl = '_'.join(['npi', str(npi), selector, reg, 'b' ,str(b), 'wd' ,str(wd), 's', str(s), 'a', str(A), '.txt']) 
        print('\n' + ttl)
        
        RT, empirical = simulate(selector, b,s,wd, A,sample_post, sample_other, prior_as_start, npi=npi)

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
# df, naned = load_fits()

#%%

wdf = df.copy()



#%%

tests = np.asarray(['conflict', 'agreement','goal', 'habit'],dtype="object")
wdf = df.query("regime == 'like_prior1' | regime == 'post_prior1'").reset_index()
conf_diff = np.zeros(wdf.shape[0])
goal_diff = np.zeros(wdf.shape[0])
agr_mean = np.zeros(wdf.shape[0])
for ind, row in wdf.iterrows():
    
    means = row['stats']['mean']
    agr_mean[ind] = means[0]
    conf_diff[ind] = (means[3] - means[1])
    goal_diff[ind]= (means[3] - means[2])

wdf['conf_diff'] = conf_diff
wdf['goal_diff'] = goal_diff
wdf['agr_mean'] = agr_mean

wdf = wdf[wdf['agr_mean'] > 100]


grouped = wdf.groupby(['npi','selector', 'regime'])['post_fit'].transform('min')
wdf['best_fit'] = grouped
wdf['optimal'] = wdf['best_fit'] == wdf['post_fit']

best = wdf[wdf['optimal']  == 1].sort_values(by=['npi', 'selector'])
titles_close = []
for ind, row in best.iterrows():    
    if row['npi'] == 3:
        data = load_file(os.getcwd() + '/parameter_data/' + row['file_ttl'])
        print(row['file_ttl'])
        print(row['stats'])
        print(row['goal_diff'])
        print(row['conf_diff'])
        titles_close.append(row['file_ttl'])

        rt_df = {
            'rts': data['RT'].ravel() ,
            'mode': tests.repeat(1000)
            }
        
        rt_df = pd.DataFrame(rt_df)
    
        plt.figure()
        sns.histplot(data=rt_df, x='rts', hue='mode')
        plt.title(str(row['avg_fit']))


#%%

pols = [3]
bs = [3]
As = [1]
ws = [1]
ss = [0.0004]
ws = [1]
selectors = ['rdm', 'ardm']  #

par_list = []
pars = [params_list[1]] + [params_list[3]] 
for p in itertools.product(pols, selectors, bs, ws, ss, As, pars):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]] + [p[5]] + [p[6]])


for title in titles_close:
    par_list.append(extract_params(title))
    

with open('sim_params.txt', 'wb') as fp:
    pickle.dump(par_list, fp)
    

#%%
pols = [3]
bs = [3]
As = [0.8]
ws = [1]
ss = [0.0034]
# ss = [0.0012]
ws = [1]
selectors = ['rdm']

p = [True, 'rdm', 3, 1, 0.0034, 1, [True, False, True, 'post_prior1']]

par_list = []
pars = [params_list[1]] 
for p in itertools.product(pols, selectors, bs, ws, ss, As, pars):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]] + [p[5]] + [p[6]])


print(par_list)
generate_data(pars)
df, fucked = load_fits_from_data(par_list)


for ind, row in df.iterrows():
    print(row.stats)

    data = load_file(path + row.file_ttl)
    rt_df = {
        'rts': data['RT'].ravel() ,
        'mode': tests.repeat(1000)
    }

    rt_df = pd.DataFrame(rt_df)
    plt.plot()
    sns.histplot(data=rt_df, x="rts", hue="mode")
    plt.show()
    plt.close()


























# %%

        
        
#%% 

'''
FIGURE NPI=3 SUCCESSFULL RT SPREAD
'''

query =  "npi == 3.0 & b == 3.0 & s == 0.0004 & w == 1.0 & A == 1.0"
succesfull = df.query(query)

plot = [1,3,2,4]        
fig,axes = plt.subplots(2,2, figsize=(10,6),sharey=True)

regimes = ["'post_prior1'", "'like_prior1'"]
selectors =["'rdm'","'ardm'"]

plt.subplots_adjust(hspace=0.4)

for sel in range(2):
    for reg in range(2):
        print(sel,reg)
        query = "npi == 3.0 & b == 3.0 & s == 0.0004 & w == 1.0 & A == 1.0 " + \
                '& selector == ' + selectors[sel] + ' & regime == ' + regimes[reg]
        sns.histplot(ax = axes[reg, sel], \
                     data=succesfull.query(query),
                     x='rts', hue='mode', legend=False, bins=70)
        axes[reg, sel].set_xlabel('RTs')
axes[0,0].set_title('RDM')
axes[0,1].set_title('ARDM')
axes[sel,reg].legend(labels = ['habit','goal', 'agreement','conflict'], bbox_to_anchor=(-0.1, -0.4), loc='lower center', ncol=4)

axes = axes.flat
for n, ax in enumerate(axes):

    ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=15, weight='bold')
        
        
#%%

''''FIGURE EXAMPLE POSTERIOR'''
query =  "npi == 3.0 & b == 3.0 & s == 0.0004 & w == 1.0 & A == 1.0"
succesfull = df.query(query)

plot = [1,3,2,4]        


regimes = ["'post_prior1'", "'like_prior1'"]
selectors =["'rdm'","'ardm'"]
labels_gray = ['posterior', '_nolegend_', '_nolegend_', '_nolegend_']
fig,axes = plt.subplots(2,2, figsize=(10,6),sharey=True)
plt.subplots_adjust(hspace=0.4)
labels_emp = ['conflict', 'agreement','goal', 'habit']
for sel in range(2):
    for reg in range(2):
        query = "npi == 3.0 & b == 3.0 & s == 0.0004 & w == 1.0 & A == 1.0 " + \
                '& selector == ' + selectors[sel] + ' & regime == ' + regimes[reg]
        d = succesfull.query(query)
        test(d)
        path = os.getcwd() + '/parameter_data/' 
        ttl = np.unique(d.file_ttl)[0]
        data = load_file(path+ttl)
        empirical = data['empirical']
        posts, x_positions = extract_post()
        
        for m in range(4):
            post = posts[m,:]
            x_pos = x_positions[m]
            axes[reg, sel].bar(x_pos, post, alpha=0.5, color='k', label= labels_gray[m])
            axes[reg, sel].bar(x_pos, empirical[m,:], alpha=0.5, color=cols[m], label = labels_emp[m])
        axes[reg,sel].get_xaxis().set_visible(False)

axes[0,0].set_title('RDM')
axes[0,1].set_title('ARDM')

axes[sel,reg].legend( bbox_to_anchor=(-0.1, -0.4), loc='lower center', ncol=5)


axes = axes.flat
for n, ax in enumerate(axes):

    ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=15, weight='bold')

plt.savefig('posteriors_example.png', dpi=300)

# # set the spacing between subplots
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)
        
        

        
#%%
query =  "npi == 3.0 & b == 3.0 & s == 0.0004 & w == 1.0 & A == 1.0"

pols = [3]
bs = [3]
As = [1]
ws = [1]
ss = [0.0004]
ratios =np.arange(2500, 1000, -100)
ws = ratios*ss[0].tolist()
selectors = ['rdm', 'ardm']  #

par_list = []
pars = [params_list[1]] + [params_list[3]] 
for p in itertools.product(pols, selectors, bs, ws, ss, As, pars):
        par_list.append([p[0]]+[p[1]] + [p[2]]+ [p[3]]+ [p[4]] + [p[5]] + [p[6]])
        
for p in par_list:
    print(p)
        
# df.query(query)

#%%
'''
GET STATISTICS OF REACTION TIME DISTRIBUTIONS AND LOOK FOR HABIT RT SHIFTS
'''


# grouped = df.groupby(['npi','selector', 'regime']).min('post_fit').reset_index()

grouped = df.groupby(['npi','selector', 'regime','b', 's', 'w','A', 'mode']).agg(
    mean = ("rts","mean"),
    median = ("rts", "median"),
    var = ("rts", "var"),
    skew =("rts", "skew"),
    post_fit = ("post_fit", 'mean'),
)

grouped = grouped.sort_values(by=['npi','selector', 'regime','b', 's', 'w','A', 'mode']).round(2).reset_index()
grouped['run'] = np.arange(0,grouped.shape[0]/4).repeat(4)


conf_diff = []
goal_diff = []

for run in np.unique(grouped['run']):
    slice = grouped.query('run == ' + str(run))
    means = slice['mean'].values
    conf_diff += [means[3] - means[1]]*4
    goal_diff += [means[3] - means[2]]*4
    
grouped['conf_diff'] = conf_diff
grouped['goal_diff'] = goal_diff


grouped = grouped.query('conf_diff < -5 & goal_diff < -5')

qs = []
for ind, row in grouped.iterrows():
    names = row.keys()[:7]
    vals = row.values[:7]
    vals
    
    for ind, val in enumerate(vals):
        if type(val) == str:
            vals[ind] = "'"+val+"'"          

    query = ' & '.join((names + ' == ' + vals.astype('str')).to_list())
    qs.append(query)

qs = set(qs)
print(qs)

    
#%%

for query in qs:
    # if((not query.__contains__('ardm')) and query.__contains__('npi == 81')):
    if(query.__contains__('ardm') and query.__contains__('npi == 8.0')):
        q = df.query(query)
        q.to_excel('poop.xlsx')
        print(query)
        print(np.unique(q.post_fit))
        if(not q.shape[0] == 4000):
            raise ValueError('WRONG QUERY SELECTION')
        plot_table(q)
        plot_figure(q,query)
        

























#%%



df = load_data_from_ttl(par_list)

#%%
df = load_data_from_ttl(par_list)
names = df.keys()


grouped = df.groupby(['npi','selector', 'regime','b', 's', 'w','A', 'mode']).agg(
    mean = ("rts","mean"),
    median = ("rts", "median"),
    var = ("rts", "var"),
    skew =("rts", "skew"),
    post_fit = ("post_fit", 'mean'),
)

grouped = grouped.sort_values(by=['npi','selector', 'regime','b', 's', 'w','A', 'mode']).round(2).reset_index()
grouped['run'] = np.arange(0,grouped.shape[0]/4).repeat(4)


conf_diff = []
goal_diff = []

for run in np.unique(grouped['run']):
    slice = grouped.query('run == ' + str(run))
    means = slice['mean'].values
    conf_diff += [(means[3] - means[1])/means[1]]*4
    goal_diff += [(means[3] - means[2])/means[2]]*4
    
grouped['conf_diff'] = conf_diff
grouped['goal_diff'] = goal_diff


# df = df.sort_values(by=['selector', 'regime', 'avg_fit'])
# df = df.drop_duplicates(subset=names[:7]).round(4)

# grouped = grouped.query('conf_diff < -5 & goal_diff < -5')

qs = []
for ind, row in grouped.iterrows():
    names = row.keys()[:7]
    vals = row.values[:7]
    vals
    
    for ind, val in enumerate(vals):
        if type(val) == str:
            vals[ind] = "'"+val+"'"          

    query = ' & '.join((names + ' == ' + vals.astype('str')).to_list())
    qs.append(query)

# qs = set(qs)

data = grouped.query("selector == 'rdm'")
# sns.lineplot(data=data, x="w", y="post_fit", hue='regime')
sns.lineplot(data=data, x="w", y="goal_diff", hue='regime')
# sns.lineplot(data=data, x="w", y="conf_diff", hue='regime')

# for query in qs:
#     # if((not query.__contains__('ardm')) and query.__contains__('npi == 81')):
#     if(not query.__contains__('ardm') and query.__contains__('npi == 3.0')):
#         q = df.query(query)
#         q.to_excel('poop.xlsx')
#         print(query)
#         print(np.unique(q.post_fit))
#         if(not q.shape[0] == 4000):
#             raise ValueError('WRONG QUERY SELECTION')
#         plot_table(q)
#         plot_figure(q,query)



