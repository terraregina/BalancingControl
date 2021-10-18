#%%
from misc import extract_params
from misc_sia import generate_data, params_list, load_fits, load_file
from misc_sia import make_title, load_fits_from_data, extract_params, tests
from misc import test_vals, simulate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle as pickle
import os as os
#%%
# df = load_fits()
# a = df[0]
# conf_mean_diff = np.zeros(a.shape[0])
# goal_mean_diff = np.zeros(a.shape[0])
# conf_med_diff = np.zeros(a.shape[0])
# goal_med_diff = np.zeros(a.shape[0])
# agr_mean = np.zeros(a.shape[0])
# goal_mean = np.zeros(a.shape[0])

# for ind, row in a.iterrows():
    
#     means = row['stats']['mean']
#     agr_mean[ind] = means[0]
#     goal_mean[ind] = means[2]

#     conf_mean_diff[ind] = ((means[3] - means[1])/means[3])
#     goal_mean_diff[ind]= ((means[3] - means[2])/means[3])

    
#     meds = row['stats']['median']
#     conf_med_diff[ind] = ((meds[3] - meds[1])/meds[3])
#     goal_med_diff[ind]= ((meds[3] - meds[2])/meds[3])

# a['conf_mean_diff'] = conf_mean_diff
# a['goal_mean_diff'] = goal_mean_diff
# a['conf_med_diff'] = conf_med_diff
# a['goal_med_diff'] = goal_med_diff
# a['agr_mean'] = agr_mean
# a['goal_mean'] = goal_mean

# #%%

# wdf = a
# # wdf = wdf[wdf['agr_mean'] > 10]
# wdf = wdf[(wdf['agr_mean'] > 15) & (wdf['agr_mean'] < 60)]


#%% BEST FITS

# wdf
# wdf['best_fit'] = wdf.groupby(['npi', 'selector','regime'])['avg_fit'].transform('min')
# wdf['optimal'] = wdf['avg_fit'] == wdf['best_fit']
# selection = wdf[(wdf['optimal'] == 1) & (wdf['selector'] == 'rdm') & (wdf['npi'] == 3)]

# print(selection)
# titles = []

# for ind, row in selection.iterrows():

     # titles.append(row['file_ttl'])
     # p = extract_params(row['file_ttl']) 
     # ttl = row['file_ttl']
     # titles.append(p)
     
     # print(row['file_ttl'])  
     # print(row['avg_fit'])
     # print(row['stats'])
     # show_rts_from_ttl(row['file_ttl'])


#%%

def calc_penalty(fit, agr_median, desired_median = 300, m=100, k=0.165, alpha=0, beta=1):
    
    median_penalty = ((agr_median - desired_median)/m)**2 + 1
    post_fit_penalty = np.exp(fit/k)

    return [alpha*median_penalty + beta*post_fit_penalty, median_penalty, post_fit_penalty]

polss = np.asarray([3,8,81,2])
modes = np.asarray(['conflict','agreement','goal','habit'])

def find_best_params(initial_params, opt_wd = False, opt_s= False, opt_a = False, opt_b = False,no = 2, iters=100,tol= 1e-3):
    npi = initial_params[0]
    selector = initial_params[1]
    b = initial_params[2]
    wd = initial_params[3]
    s = initial_params[4]
    a = initial_params[5]
    sample_post, sample_other, prior_as_start, reg = initial_params[6]
    
    ind = np.where(polss == npi)[0][0]
    posts = np.zeros([len(modes), npi])      # translate the posteriors
    post = np.asarray(test_vals)[ind,:,0]  # into a numpy array

    for indx, p in enumerate(post):
        posts[indx,:] = np.asarray(p)
    
    best_fit = np.infty
    i = 0

    post_fits = np.zeros([iters, no+1])
    fit_penalties = np.zeros([iters, no+1])
    agr_meds = np.zeros([iters, no+1])
    med_penalties = np.zeros([iters, no+1])
    penalties = np.zeros([iters, no+1])
    best_wds = np.zeros(iters)
    best_ass = np.zeros(iters)
    best_bs = np.zeros(iters)
    best_ss = np.zeros(iters)
    # calc fitness of initial guess
    np.random.seed(0)
    RT, empirical = simulate(selector, b, s, wd, a, sample_post, sample_other, prior_as_start, npi=npi)
     
    post_fits[i,0] = np.abs((posts - empirical)/posts).mean(axis=1).mean()
    agr_meds[i,0] = np.median(RT[1,:])
    penalties[i,0], med_penalties[i,0], fit_penalties[i,0] = calc_penalty(post_fits[i,0], agr_meds[i,0])
 
    best_fit_counter = 0
    total = 0
    last_best_fit = 1042342

    factor = 0.15
    while(best_fit > tol and i < iters and total < 30):

        np.random.seed()
        if opt_b:
            bs = np.append(np.array([b]), np.random.normal(b,b*factor,no))
        else:
            bs = np.append(np.array([b]), np.random.normal(b,0,no))

        if opt_wd:
            wds = (wd/s+ np.append(np.array([0]), np.random.normal(0,wd/s*factor, no)))*s
        else:
            wds = np.append(np.array([wd]), np.random.normal(wd, 0, no))


        # ss = np.append(np.array([s]), np.random.normal(s, s*0.05, no))
        if opt_s:
            ss = np.append(np.array([s]), np.random.normal(s, s*factor, no))
        else:
            ss = np.append(np.array([s]), np.random.normal(s, 0, no))
     
        if opt_a:
            ass = np.append(np.array([a]), np.random.normal(a, a*factor, no))
        else:
            ass = np.append(np.array([a]), np.random.normal(a, 0, no))
     
        # calc offspring fitness
        for o in range(no):

            np.random.seed(0)
            RT, empirical = simulate(selector, bs[o+1], ss[o+1], wds[o+1], ass[o+1], sample_post, sample_other, prior_as_start, npi=npi)

            post_fits[i,o+1] = np.abs((posts - empirical)/posts).mean(axis=1).mean()
            agr_meds[i,o+1] = np.median(RT[1,:])
            penalties[i,o+1], med_penalties[i,o+1], fit_penalties[i,o+1] = calc_penalty(post_fits[i,o+1], agr_meds[i,o+1])

        # select best candidate
        if opt_b:
            print(bs)
        elif opt_wd:
            print(wds)
        elif opt_a:
            print(ass)
        elif opt_s:
            print(ss)

        print("\npost_fits", post_fits[i,:])
        # print("fit_penalties", fit_penalties)
        # print("med_penalties", med_penalties)
        # print("penalties", penalties)
        best = np.argmin(penalties[i,:])
        b, bs[best] = [bs[best]]*2
        wd, wds[best] = [wds[best]]*2
        s, ss[best] = [ss[best]]*2
        a, ass[best] = [ass[best]]*2
      
        best_wds[i] = wd
        best_ss[i] = s
        best_ass[i] = a
        best_bs[i] = b

        post_fits[i+1,0] = post_fits[i,best]
        fit_penalties[i+1,0] = fit_penalties[i,best]
        agr_meds[i+1,0] = agr_meds[i,best]
        med_penalties[i+1,0] = med_penalties[i,best]
        penalties[i+1,0] = penalties[i,best]
        best_fit = post_fits[i,best]
        print(i, best, best_fit)
        print(b,wd,s,a)

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
        'best_ass': best_ass,
        'best_ss': best_ss,
        'best_bs': best_bs,
        'params': initial_params,
    }

    return results

# NEW OPTIMAL PARAMETERS
#%%
from misc_sia import show_rts, generate_data
# original
# optimal_parameters = [
#  [3, 'rdm', 1, 1.5, 0.005, 1, [False, False, True, 'standard']],
#  [3, 'rdm', 1, 1.5, 0.005, 1, [True, False, True, 'post_prior1']],
#  [3, 'rdm', 1, 2.613, 0.013000000000000001, 1, [True, False, False, 'post_prior0']],
#  [3, 'rdm', 1.9, 1.81, 0.01, 1, [False, True, True, 'like_prior1']],
#  [3, 'rdm',  1,1.5, 0.0036, 1, [False, True, False, 'like_prior0']]]

# tweek wds
# optimal_parameters = [
# [3, 'rdm', 1, 2.0368092803778026, 0.005, 1, [False, False, True, 'standard']],
# [3, 'rdm', 1, 1.241687356873713, 0.005, 1, [True, False, True, 'post_prior1']],
# [3, 'rdm', 1, 5.914959918032103, 0.013000000000000001, 1, [True, False, False, 'post_prior0']],
# [3, 'rdm', 1.9, 1.8059582685494318, 0.01, 1, [False, True, True, 'like_prior1']],
# [3, 'rdm', 1, 1.7303636466497805, 0.0036, 1, [False, True, False, 'like_prior0']]]

# tweek ss
# optimal_parameters = [
# [3, 'rdm', 1, 2.0368092803778026, 0.00523553064850987, 1, [False, False, True, 'standard']],
# [3, 'rdm', 1, 1.241687356873713, 0.004977473268097638, 1, [True, False, True, 'post_prior1']],
# [3, 'rdm', 1, 5.914959918032103, 0.013000000000000001, 1, [True, False, False, 'post_prior0']],
# [3, 'rdm', 1.9, 1.8059582685494318, 0.01, 1, [False, True, True, 'like_prior1']],
# [3, 'rdm', 1, 1.7303636466497805, 0.003673934170060218, 1, [False, True, False, 'like_prior0']]]


# for p in optimal_parameters:
    # generate_data([p])
    # show_rts([p])

# optimal_parameters_old = [
# [3, 'rdm', 1, 2.0368092803778026, 0.005, 1, [False, False, True, 'standard']],
# [3, 'rdm', 1, 1.241687356873713, 0.004977473268097638, 1, [True, False, True, 'post_prior1']],
# [3, 'rdm', 1, 5.914959918032103, 0.013000000000000001, 1, [True, False, False, 'post_prior0']],
# [3, 'rdm', 1.9, 1.8059582685494318, 0.01, 1, [False, True, True, 'like_prior1']],
# [3, 'rdm', 1, 1.7303636466497805, 0.003673934170060218, 1, [False, True, False, 'like_prior0']]]

# compare old and new 

optimal_parameters_new = [
[3, 'rdm', 1, 2.0368092803778026, 0.005, 0.85, [False, False, True, 'standard']],
[3, 'rdm', 1, 1.241687356873713, 0.005, 0.5, [True, False, True, 'post_prior1']],
[3, 'rdm', 1.1, 5.914959918032103, 0.014000000000000001, 0, [True, False, False, 'post_prior0']],
[3, 'rdm', 1.8, 2.059582685494318, 0.01, 0.35, [False, True, True, 'like_prior1']],
[3, 'rdm', 1, 1.7303636466497805, 0.0036, 1, [False, True, False, 'like_prior0']]]


ttls_for_fig = ['npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt']
                
ttls_for_fig = ['npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-06_a_3.5_.txt']

pars = []

for ttl in ttls_for_fig:
    pars.append(extract_params(ttl))

# optimal_parameters_new = [[3, 'rdm', 1.1, 5.914959918032103, 0.0064000000000000001, 0, [True, False, False, 'post_prior0']]]
optimal_parameters_new = [
[3, 'rdm', 1, 2.0368092803778026, 0.005, 0.85, [False, False, True, 'standard']],
[3, 'rdm', 1, 1.241687356873713, 0.005, 0.5, [True, False, True, 'post_prior1']],
[3, 'rdm', 1.1, 5.914959918032103, 0.014000000000000001, 0, [True, False, False, 'post_prior0']],
[3, 'rdm', 1.8, 2.059582685494318, 0.01, 0.35, [False, True, True, 'like_prior1']],
[3, 'rdm', 1, 1.7303636466497805, 0.0036, 1, [False, True, False, 'like_prior0']]]

trials=4000
for par in optimal_parameters_new:
    # generate_data([par],trials=trials)
    show_rts([par],trials=trials)
# extract optimal

# for i in range(2):
    # print(params_list[i][3])
    # fits = load_file('optimal_fit_rdm_ass_' + params_list[i][3])
    # ind = np.max(np.nonzero(fits['best_ass']))
    # print(fits['best_ass'][ind])
    # optimal_parameters_new[i][4] = fits['best_ss'][ind]


# for p in optimal_parameters_new:
#     results = find_best_params(p,opt_a=True)

#     with open(os.getcwd() + '\\optimal_fit_rdm_ass_'+  p[6][3], 'wb') as fp:
#         data = pickle.dump(results, fp)





#%%
# optimal_parameters = [[3, 'rdm', 1, 1.7303636466497805, 0.0036, 1, [False, True, False, 'like_prior0']],
#  [3, 'rdm', 1.9, 1.52630485, 0.01, 1, [False, True, True, 'like_prior1']],
#  [3, 'rdm', 1, 3.77675973, 0.013000000000000001, 1, [True, False, False, 'post_prior0']],
#  [3, 'rdm', 1, 1.34164878, 0.005, 1, [True, False, True, 'post_prior1']],
#  [3, 'rdm', 1, 1.75, 0.005, 1, [False, False, True, 'standard']]]

# p = optimal_parameters[0]
# p = [3, 'rdm', 1, 2, 0.00036100000000000026, 1, [False, False, True, 'standard']]
# generate_data([p])
# show_rts([p])
# def show_rts_from_ttl(ttl):

#     data = load_file(os.getcwd() + '/parameter_data/' + ttl)
#     rt_df = {
#         'rts': data['RT'].ravel() ,
#         'mode': tests.repeat(1000)
#         }
    
#     rt_df = pd.DataFrame(rt_df)
#     sns.histplot(data=rt_df, x='rts', hue='mode',bins=150)
#     plt.show()
