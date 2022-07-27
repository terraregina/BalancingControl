
# %%
# # %%
# Imports
# 

import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import chart_studio.plotly as py
# import plotly.express as px
import pandas as pd
# import cufflinks as cf
import json as js
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)
from itertools import product, repeat
import os
import action_selection as asl
from itertools import product
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import torch as ar
import perception as prc
import agent as agt
from environment import PlanetWorld
from agent import BayesianPlanner
from world import World
from planet_sequences import generate_trials_df
import time
import time
from multiprocessing import Pool
import multiprocessing.pool as mpp
import tqdm
from run_exampe_habit_v1 import run_agent

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def run_single_sim(lst,
                    ns,
                    na,
                    npl,
                    nc,
                    nr,
                    T,
                    state_transition_matrix,
                    planet_reward_probs,
                    planet_reward_probs_switched,
                    repetitions, use_fitting):


    switch_cues, contingency_degradation, reward_naive, context_trans_prob, cue_ambiguity, h,\
    training_blocks, degradation_blocks, trials_per_block, dec_temp, rew, util, config_folder = lst
    
    config = 'planning_config' + '_degradation_'+ str(int(contingency_degradation)) \
                      + '_switch_' + str(int(switch_cues))                \
                      + '_train' + str(training_blocks)                   \
                      + '_degr' + str(degradation_blocks)                 \
                      + '_n' + str(trials_per_block)+'.json'


    folder = os.path.join(os.getcwd(),'config/' + config_folder)
    file = open(os.path.join(folder,config))

    task_params = js.load(file)                                                                                 
    colors = np.asarray(task_params['context'])          # 0/1 as indicator of color
    sequence = np.asarray(task_params['sequence'])       # what is the optimal sequence
    starts = np.asarray(task_params['starts'])           # starting position of agent
    planets = np.asarray(task_params['planets'])         # planet positions 
    trial_type = np.asarray(task_params['trial_type'])
    blocks = np.asarray(task_params['block'])


    nblocks = int(blocks.max()+1)         # number of blocks
    trials = blocks.size                  # number of trialsor
    block = task_params['trials_per_block']           # trials per block
 
    meta = {
        'trial_file' : config, 
        'trial_type' : trial_type,
        'switch_cues': switch_cues == True,
        'contingency_degradation' : contingency_degradation == True,
        'learn_rew' : reward_naive == True,
        'context_trans_prob': context_trans_prob,
        'cue_ambiguity' : cue_ambiguity,
        'h' : h,
        'optimal_sequence' : sequence,
        'blocks' : blocks,
        'trials' : trials,
        'nblocks' : nblocks,
        'degradation_blocks': task_params['degradation_blocks'],
        'training_blocks': task_params['training_blocks'],
        'interlace': task_params['interlace'],
        'contingency_degrdataion': task_params['contingency_degradation'],
        'switch_cues': task_params['switch_cues'],
        'trials_per_block': task_params['trials_per_block']
    }

    all_optimal_seqs = np.unique(sequence)                                                                            

    # reward probabilities schedule dependent on trial and planet constelation
    Rho = np.zeros([trials, nr, ns])

    for i, pl in enumerate(planets):
        if i >= block*meta['training_blocks'] and i < block*(meta['training_blocks'] + meta['degradation_blocks']) and contingency_degradation:
            # print(i)
            # print(pl)
            Rho[i,:,:] = planet_reward_probs_switched[tuple([pl])].T
        else:
            Rho[i,:,:] = planet_reward_probs[tuple([pl])].T

    # u = 0.99
    # utility = np.array([(1-u)/2,(1-u)/2,u])

    utility = np.array([float(u)/100 for u in util])
    # print(utility)

    if reward_naive==True:
        reward_counts = np.ones([nr, npl, nc])
    else:
        reward_counts = np.tile(planet_reward_probs.T[:,:,np.newaxis]*5,(1,1,nc))+1

    par_list = [h,                        
                context_trans_prob,
                cue_ambiguity,            
                'avg',                    
                Rho,                      
                utility,                  
                state_transition_matrix,  
                planets,                  
                starts,                   
                colors,
                reward_counts,
                1,
                dec_temp,
                rew]

    prefix = 'multiple_'

    if use_fitting == True:
        prefix += 'fitt_'
    else:
        prefix +='hier_'

    if switch_cues == True:
        prefix += 'switch1_'
    else:
        prefix +='switch0_'

    if contingency_degradation == True:
        prefix += 'degr1_'
    else:
        prefix += 'degr0_'

    fname = prefix +'p' + str(cue_ambiguity) +'_learn_rew' + str(int(reward_naive == True)) + '_q' + str(context_trans_prob) + '_h' + str(h)+ '_' +\
    str(meta['trials_per_block']) +'_'+str(meta['training_blocks']) + str(meta['degradation_blocks']) +\
    '_dec' + str(dec_temp)+ '_rew' + str(rew) + '_u' +  '-'.join(util) + '_' + config_folder
 
    fname +=  '_extinguish.json'

    worlds = [run_agent(par_list, trials, T, ns , na, nr, nc, npl, added=[trial_type,sequence], use_fitting=use_fitting) for _ in range(repetitions)]
    meta['trial_type'] = task_params['trial_type']
    meta['optimal_sequence'] = task_params['sequence']

    worlds.append(meta)

   
    fname = os.path.join(os.path.join(os.getcwd(),'temp'), fname)
    jsonpickle_numpy.register_handlers()
    pickled = pickle.encode(worlds)
    with open(fname, 'w') as outfile:
        json.dump(pickled, outfile)


    return fname

def pooled(arrays):
    use_fitting = False

    repetitions = 1

    seed = 521312
    np.random.seed(seed)
    ar.manual_seed(seed)

    lst = []
    path = os.path.join(os.getcwd(),'temp')
    existing_files = os.listdir(path)

    for i in product(*arrays):
        lst.append(list(i))


    names = []

    for li, l in enumerate(lst):
        prefix = 'multiple_'
        if l[0] == True:
            prefix += 'switch1_'
        else:
            prefix +='switch0_'

        if l[1] == True:
            prefix += 'degr1_'
        else:
            prefix += 'degr0_'


        l[11] = [str(entry) for entry in l[11]]
        fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
        str(l[8]) + '_' + str(l[6]) + str(l[7]) + \
        '_dec' + str(l[9]) +'_rew' + str(l[10]) + '_' + 'u'+  '-'.join(l[11]) + '_' + l[12]

        if extinguish:
            fname += '_extinguish.json'
        else:
            fname += '.json'
        names.append([li, fname])


    print('simulations to run: ' + str(len(lst)))

    ca = [ns, na, npl, nc, nr, T, state_transition_matrix, planet_reward_probs,\
        planet_reward_probs_switched,repetitions,use_fitting]

    with Pool() as pool:

        for _ in tqdm.tqdm(pool.istarmap(run_single_sim, zip(lst,\
                                                repeat(ca[0]),\
                                                repeat(ca[1]),\
                                                repeat(ca[2]),\
                                                repeat(ca[3]),\
                                                repeat(ca[4]),\
                                                repeat(ca[5]),\
                                                repeat(ca[6]),\
                                                repeat(ca[7]),\
                                                repeat(ca[8]),\
                                                repeat(ca[9]),\
                                                repeat(ca[10]))),
                        total=len(lst)):
            pass
    
if __name__ == '__main__':


    data_folder = 'temp'


    extinguish = True

    na = 2                                           # number of unique possible actions
    nc = 4                                           # number of contexts, planning and habit
    nr = 3                                           # number of rewards
    ns = 6                                           # number of unique travel locations
    npl = 3
    steps = 3                                        # numbe of decisions made in an episode
    T = steps + 1                                    # episode length


    planet_reward_probs = np.array([[0.95, 0   , 0   ],
                                    [0.05, 0.95, 0.05],
                                    [0,    0.05, 0.95]]).T    # npl x nr
    planet_reward_probs_switched = np.array([[0   , 0    , 0.95],
                                            [0.05, 0.95 , 0.05],
                                            [0.95, 0.05 , 0.0]]).T 
    state_transition_matrix = np.zeros([ns,ns,na])
    m = [1,2,3,4,5,0]
    for r, row in enumerate(state_transition_matrix[:,:,0]):
        row[m[r]] = 1
    j = np.array([5,4,5,6,2,2])-1
    for r, row in enumerate(state_transition_matrix[:,:,1]):
        row[j[r]] = 1
    state_transition_matrix = np.transpose(state_transition_matrix, axes= (1,0,2))
    state_transition_matrix = np.repeat(state_transition_matrix[:,:,:,np.newaxis], repeats=nc, axis=3)



import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import pandas as pd
import cufflinks as cf
import json as js
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import itertools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=False)
import plotly.io as pio
pio.renderers.default = 'notebook_connected'
import os
import action_selection as asl
from itertools import product, repeat
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import plotly.graph_objects as go
import perception as prc
import agent as agt
from environment import PlanetWorld
from agent import BayesianPlanner
from world import World
from planet_sequences import generate_trials_df
from multiprocessing import Pool
import time
# %matplotlib inline

# functions
def load_file_names(arrays, use_fitting=False):
    lst = []
    for i in product(*arrays):
        lst.append(list(i))
    
    names = []
    print('files to load: ' + str(len(lst)))
    for li, l in enumerate(lst):

        prefix = 'multiple_'
        if use_fitting == True:
            prefix += 'fitt_'
        else:
            prefix +='hier_'

        if l[0] == True:
            prefix += 'switch1_'
        else:
            prefix +='switch0_'

        if l[1] == True:
            prefix += 'degr1_'
        else:
            prefix += 'degr0_'

        # fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
        # str(l[8]) + '_' + str(l[6]) + str(l[7])+ '_dec' + str(l[9])
        
        l[11] = [str(entry) for entry in l[11]]
        fname = prefix + 'p' + str(l[4])  +'_learn_rew' + str(int(l[2] == True))+ '_q' + str(l[3]) + '_h' + str(l[5]) + '_' +\
        str(l[8]) + '_' + str(l[6]) + str(l[7]) + \
        '_dec' + str(l[9]) +'_rew' + str(l[10]) + '_' + 'u'+  '-'.join(l[11]) + '_' + l[12]

        # # print(len(l))
        # if len(l) > 10:
        #     fname += '_' + l[-1]

        fname +=  '_extinguish.json'

        names.append(fname)


    return names

def load_df(names,data_folder='data', extinguish=None):

    if extinguish is None:
        raise('did not specify if rewarded during extinction')
    # if not just_simulated:
    path = os.path.join(os.getcwd(),data_folder)
    #     names = os.listdir(path)
    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)
        # print(names[fi])

    dfs = [None]*len(names)

    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()
        with open(fname, 'r') as infile:
            data = json.load(infile)
        worlds = pickle.decode(data)
        meta = worlds[-1]
        agents = [w.agent for w in worlds[:-1]]
        perception = [w.agent.perception for w in worlds[:-1]]
        nt = worlds[0].T
        npl = perception[0].npl
        nr = worlds[0].agent.nr
        nc = perception[0].nc
        nw = len(worlds[:-1])
        ntrials = meta['trials']
        learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
        switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
        contingency_degradation = np.repeat(meta['contingency_degradation'], ntrials*nw*nt)
        tr_per_block = np.repeat(meta['trials_per_block'], ntrials*nw*nt)
        ndb = np.repeat(meta['degradation_blocks'], ntrials*nw*nt)
        ntb = np.repeat(meta['training_blocks'], ntrials*nw*nt)
        post_dir_rewards = [a.posterior_dirichlet_rew for a in agents]
        post_dir_rewards = [post[:,1:,:,:] for post in post_dir_rewards]
        entropy_rewards = np.zeros([nw*ntrials*nt,nc])
        extinguished = np.zeros(ntrials*nw*nt, dtype='int32')
        extinguished[:] = int(extinguish == True)
        prior_rewards = worlds[0].agent.perception.prior_rewards
        utility_0 = np.repeat(prior_rewards[0], ntrials*nw*nt)
        utility_1 = np.repeat(prior_rewards[1], ntrials*nw*nt)
        utility_2 = np.repeat(prior_rewards[2], ntrials*nw*nt)
        r_lambda = np.repeat(worlds[0].agent.perception.r_lambda, ntrials*nw*nt)

        for ip, post in enumerate(post_dir_rewards):
            post = post.sum(axis=3)                      # sum out planets
            norm = post.sum(axis=2)                      # normalizing constants
            reward_distributions = np.zeros(post.shape)

            for r in range(nr):                          # normalize each reward 
                reward_distributions[:,:,r,:] = np.divide(post[:,:,r,:],norm)
            entropy = np.zeros([ntrials, nt, nc])

            for trl in range(ntrials):
                for t in range(nt-1):
                    prob = reward_distributions[trl,t,:,:].T
                    # if prob.sum() == 0:
                    #     print('problem')
                    entropy[trl,t+1,:] = -(np.log(prob)*prob).sum(axis=1)

            entropy[:,0,:] = None
            entropy_rewards[ip*(ntrials*nt):(ip+1)*(ntrials*nt),:] = np.reshape(entropy, [ntrials*nt,nc])

        entropy_context = np.zeros(ntrials*nt*nw)
        post_context = [a.posterior_context for a in agents]

        for ip, post in enumerate(post_context):
            entropy = np.zeros([ntrials, nt])

            for trl in range(ntrials):
                entropy[trl,:] = -(np.log(post[trl,:])*post[trl,:]).sum(axis=1) 
            entropy_context[ip*(ntrials*nt):(ip+1)*(ntrials*nt)] = np.reshape(entropy, [ntrials*nt])

        posterior_context = [agent.posterior_context for agent in agents]
        observations = [w.observations for w in worlds[:-1]]
        context_cues = worlds[0].environment.context_cues
        policies = worlds[0].agent.policies
        actions = [w.actions[:,:3] for w in worlds[:-1]] 
        true_optimal = np.tile(np.repeat(meta['optimal_sequence'],nt), nw)
        cue = np.tile(np.repeat(context_cues, nt), nw)
        ex_p = np.zeros(ntrials)
        executed_policy = np.zeros(nw*ntrials,dtype='int32')
        optimality = np.zeros(nw*ntrials)
        chose_optimal = np.zeros(nw*ntrials)

        for w in range(nw):
            for pi, p in enumerate(policies):
                inds = np.where( (actions[w][:,0] == p[0]) & (actions[w][:,1] == p[1]) & (actions[w][:,2] == p[2]) )[0]
                ex_p[inds] = pi
            executed_policy[w*ntrials:(w+1)*ntrials] = ex_p
            ch_op = executed_policy[w*ntrials:(w+1)*ntrials] == meta['optimal_sequence']
            chose_optimal[w*ntrials:(w+1)*ntrials] = ch_op
            optimality[w*ntrials:(w+1)*ntrials] = np.cumsum(ch_op)/(np.arange(ntrials)+1)

        executed_policy = np.repeat(executed_policy, nt)
        chose_optimal = np.repeat(chose_optimal, nt)
        optimality = np.repeat(optimality, nt)
        no = perception[0].generative_model_context.shape[0]
        optimal_contexts = [np.argmax(perception[0].generative_model_contexts[i,:] for i in range(no))]
        true_context = 0
        q = np.repeat(meta['context_trans_prob'], ntrials*nw*nt)
        p = np.repeat(meta['cue_ambiguity'], ntrials*nw*nt)
        h = np.repeat(meta['h'], ntrials*nw*nt)
        dec_temp = np.repeat(worlds[0].dec_temp,ntrials*nw*nt)
        switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
        learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
        degradation = np.repeat('contingency_degradation', ntrials*nw*nt)
        trial_type = np.tile(np.repeat(meta['trial_type'], nt), nw)
        trial = np.tile(np.repeat(np.arange(ntrials),nt), nw)
        run = np.repeat(np.arange(nw),nt*ntrials)
        run.astype('str')
        inferred_context_t0 = np.zeros(ntrials*nw,dtype='int32')
        inferred_context_t3  = np.zeros(ntrials*nw,'int32')
        agnt = np.repeat(np.arange(nw)+f*nw,nt*ntrials)
        true_context = np.zeros(ntrials*nw, dtype='int32')
        no, nc = perception[0].generative_model_context.shape 
        modes_gmc =  perception[0].generative_model_context.argsort(axis=1)
        contexts = [modes_gmc[i,:][-2:] for i in range(no)] # arranged in ascending order!
        if_inferred_context_switch = np.zeros(ntrials, dtype="int32")

        # c is an array holding what contexts should be for the given task trials [0,1,0,1,..2,3,2,3...0,1] etc
        to = np.zeros(ntrials, dtype="int32")
        for i in range(no):    
            c = np.array([contexts[i][-1]]*(meta['trials_per_block']*meta['training_blocks'])\
                + [contexts[i][-2]]*(meta['trials_per_block']*meta['degradation_blocks'])\
                + [contexts[i][-1]]*meta['trials_per_block']*2)
            to[np.where(context_cues == i)] = c[np.where(context_cues == i)]
            # print(to)
            if_inferred_context_switch[np.where(context_cues == i)] = c[np.where(context_cues == i)]
        inferred_switch = np.zeros(ntrials*nw,dtype='int32')
        context_optimality = np.zeros(ntrials*nw)

        for w in range(nw):
            inferred_context_t0[w*ntrials:(w+1)*ntrials] = np.argmax(posterior_context[w][:,0,:],axis=1)
            inferred_context_t3[w*ntrials:(w+1)*ntrials] = np.argmax(posterior_context[w][:,-1,:],axis=1)
            inferred_switch[w*ntrials:(w+1)*ntrials] = if_inferred_context_switch == \
                                                    inferred_context_t3[w*ntrials:(w+1)*ntrials]
            context_optimality[w*ntrials:(w+1)*ntrials] = np.cumsum(inferred_switch[w*ntrials:(w+1)*ntrials])\
                                                                /(np.arange(ntrials)+1)
        true_context[w*ntrials:(w+1)*ntrials] = to
        inferred_switch = np.repeat(inferred_switch, nt)
        inferred_context_t0 = np.repeat(inferred_context_t0, nt)
        inferred_context_t3 = np.repeat(inferred_context_t3, nt)
        context_optimality = np.repeat(context_optimality, nt)
        true_context =  np.repeat(true_context, nt)
        t = np.tile(np.arange(4), nw*ntrials)
        
        # print(true_context.size)
        # print(trial_type.size)
        d = {'trial_type':trial_type, 'run':run, 'trial':trial, 't':t, 'true_optimal':true_optimal,\
                            'cue':cue, 'q':q, 'p':p, 'h':h, 'inferred_context_t0':inferred_context_t0,\
                            'inferred_context_t3':inferred_context_t3, 'executed_policy':executed_policy,\
                            'chose_optimal': chose_optimal, 'entropy_rew_c1': entropy_rewards[:,0], 'entropy_rew_c2': entropy_rewards[:,1], \
                            'entropy_rew_c3': entropy_rewards[:,2] , 'entropy_rew_c4': entropy_rewards[:,3],\
                            'policy_optimality':optimality,'agent':agnt, 'inferred_switch': inferred_switch,\
                            'context_optimality':context_optimality, 'learn_rew': learn_rew, 'entropy_context':entropy_context, \
                            'switch_cues':switch_cues, 'contingency_degradation': contingency_degradation,\
                            'degradation_blocks': ndb, 'training_blocks':ntb, 'trials_per_block': tr_per_block,\
                            'true_context': true_context, 'dec_temp':dec_temp, 'utility_0': utility_0, 'utility_1': utility_1, 'utility_2': utility_2,
                            'r_lambda': r_lambda} 

        # for key in d.keys():
        #     print(key, np.unique(d[key].shape))
        dfs[f] = pd.DataFrame(d)
        
    data = pd.concat(dfs)

    groups = ['agent','run', 't','degradation_blocks', 'training_blocks', 'trials_per_block','learn_rew', 'p', 'q','h','cue']
    grouped = data.groupby(by=groups)
    data['iterator'] = 1
    data['ith_cue_trial'] = grouped['iterator'].transform('cumsum')
    data['policy_optimality_cue'] = grouped['chose_optimal'].transform('cumsum') / data['ith_cue_trial']
    data['context_optimal_cue'] = grouped['inferred_switch'].transform('cumsum') / data['ith_cue_trial']
    data.drop('iterator',1,inplace =True)
    data.astype({'h': 'category'})

    return data


def context_plot(query='p == 0.6'):

    # context and policy optimality
    fig = plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    print('cue')
    plot_df = base_df.query(query + '& t ==' + str(t) + ' & cue == ' + str(cue))
    plot_df['h'] = plot_df['h'].astype('category')
    ax = sns.lineplot(data=plot_df, x='trial', y='context_optimality', hue='h',\
                      palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size), legend=False)
    ax.set(ylim = (0,1.1))
    ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
    cols = [[1,1,1], [0,0,0],[1,1,1]] 
    for i, row in ranges.iterrows():
        ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
    # reward distribution entropy for different contexts
    plt.subplot(1,2,2)
    ax = sns.lineplot(data=plot_df, x='trial', y='policy_optimality', hue='h',\
            palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size))
    ax.legend(ncol = np.unique(plot_df['h']).size, bbox_to_anchor=(-2, -0.25), loc='upper left',\
              borderaxespad=0,title='h')
    cols = [[1,1,1], [0,0,0],[1,1,1]] 
    for i, row in ranges.iterrows():
        ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)

    ax.set(ylim = (0,1))
    # title = base_query + ' & ' + query
    # title = title.replace(' & ', ', ')
    # title = title.replace('==', ':')
    title = 'Inferred context and policy optimality for p: ' + query[-3:] + ', switch cues: '\
            + str(int(switch)) + ', degradation: ' + str(int(contingency_degr)) + \
            ', reward_naive: ' + str(int(reward_naive)) + ', cue_shown: ' + str(cue)
    fig.suptitle(title, fontsize=15)



def reward_entropy_plot(query='p == 0.6'):

    # entropy of reward distribution for each context 
    plot_df = base_df.query(query + '& t ==' + str(t) + ' & cue == ' + str(cue))
    plot_df['h'] = plot_df['h'].astype('category')
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))
    axs_flat = axs.flatten()
    ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])

    legs = [False, False, False, True]
    ys = ['entropy_rew_c1', 'entropy_rew_c2','entropy_rew_c3','entropy_rew_c4',]
    for c in range(nc):
        sns.lineplot(ax=axs_flat[c], data=plot_df, x='trial',y=ys[c],hue='h',\
            palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size), legend=legs[c])
        cols = [[1,1,1], [0,0,0],[1,1,1]]
        for i, row in ranges.iterrows():
            axs_flat[c].axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
        axs_flat[-1].legend(ncol = np.unique(plot_df['h']).size, bbox_to_anchor=(-2, -0.25), loc='upper left',\
                            borderaxespad=0.5,title='h')
    
    title = 'Reward distribution entropy for  p: ' + query[-3:] + ', switch cues: ' + str(int(switch)) +\
            ', degradation: ' + str(int(contingency_degr)) + ', reward_naive: ' + str(int(reward_naive)) + \
            ', cue_shown: ' + str(cue)
    fig.suptitle(title, fontsize=15)

def context_plot_cue_dependent(query='p == 0.6',print_counts=False,util = None, save=False):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,6), sharex = True)

    lgnd = [False, True]
    cues = [0,1]
    titles_c = ['Inferred context optimality for cue 0','Inferred context optimality (at t4) for cue 1']
    titles_p = ['Inferred policy optimality for cue 0','Inferred policy optimality for cue 1']

    for cue in cues:
        plot_df = base_df.query(query + '& t ==' + str(t) + ' & cue == ' + str(cue))

        sns.lineplot(ax = axes[0,cue], data=plot_df, x='trial', y='context_optimal_cue', hue='h',\
                      palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size),legend=False)
        axes[0,cue].set_title(titles_c[cue])
        axes[1,cue].set_title(titles_p[cue])
        sns.lineplot(ax=axes[1,cue], data=plot_df, x='trial', y='policy_optimality_cue', hue='h',\
            palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size), legend=lgnd[cue])
    axes[1,1].legend(ncol = np.unique(plot_df['h']).size, bbox_to_anchor=(-2, -0.25), loc='upper left',\
              borderaxespad=0,title='h')

    if print_counts:
        test_df =  base_df.query(query + '& t ==' + str(t))
        test_df['correct'] = test_df['inferred_context_t3']== test_df['true_context']
        cols = ['agent','h', 'trial','trial_type', 'cue', 'inferred_context_t3', 'true_context','correct']
        counts = test_df[cols].groupby(by=['h','trial_type','cue','correct']).size()
        print(counts)

    ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
    cols = [[1,1,1], [0,0,0],[1,1,1]] 
    for ax in axes.flatten():
        ax.set_ylim([0,1.05])
        for i, row in ranges.iterrows():
            ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
    


    # title = base_query + ' & ' + query
    # title = title.replace(' & ', ', ')
    # title = title.replace('==', ':')
    title = 'p: ' + query[-3:] + ', switch cues: '\
            + str(int(switch)) + ', degradation: ' + str(int(contingency_degr)) + \
            ', reward_naive: ' + str(int(reward_naive)) + ' ' + util
    fig.suptitle(title, fontsize=15)
    if save:
        return fig


def load_df_animation_context(names,data_folder='temp'):

    path = os.path.join(os.getcwd(),data_folder)
    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)


    overall_df = [None for _ in range(len(names))]
    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            data = json.load(infile)

        worlds = pickle.decode(data)
        meta = worlds[-1]
        nw = len(worlds[:-1])
        agents = [w.agent for w in worlds[:-1]]
        posterior_context = [agent.posterior_context for agent in agents]
        ntrials, t, nc = posterior_context[0].shape
        outcome_surprise = [agent.outcome_suprise for agent in agents]
        policy_surprise = [agent.policy_surprise for agent in agents]
        policy_entropy = [agent.policy_entropy for agent in agents]
        npi = agents[0].posterior_policies.shape[2]
        taus = np.arange(ntrials)
        ts = np.arange(t)
        cs = np.arange(nc)
        pis_post = np.array(['post_' for _ in range(npi)], dtype=object) + np.array([str(i) for i in range(npi)], dtype=object)
        pis_prior = np.array(['post_' for _ in range(npi)], dtype=object) + np.array([str(i) for i in range(npi)], dtype=object)
        pis_like = np.array(['post_' for _ in range(npi)], dtype=object) + np.array([str(i) for i in range(npi)], dtype=object)

        mi = pd.MultiIndex.from_product([taus, ts, cs], names=['trial', 't', 'context'])
        mi_post = pd.MultiIndex.from_product([taus, ts, pis_post, cs], names=['trial', 't', 'policy','context'])
        mi_like = pd.MultiIndex.from_product([taus, ts, pis_like, cs], names=['trial', 't', 'policy','context'])
        mi_prior = pd.MultiIndex.from_product([taus, ts, pis_like, cs], names=['trial', 't', 'policy','context'])

        dfs = [None for _ in range(nw)]
        factor = ntrials*nc*t

        for w in range(nw):

            policy_post_df =  pd.Series(index=mi_post, data=agents[w].posterior_policies.flatten())
            policy_post_df = policy_post_df.unstack(level='policy').reset_index()
            # fix entropy calcs
            policy_prob = policy_post_df.iloc[:,-8:].to_numpy()
            policy_prob[policy_prob == 0] = 10**(-300)
            entropy = -(policy_prob*np.log(policy_prob)).sum(axis=1)

            prior_policies = np.tile(agents[w].prior_policies[:,np.newaxis,:,:], (1,t,1,1))
            # print(np.all(prior_policies[:,0,:,:] == prior_policies[:,1,:,:] ))
            policy_prior_df =  pd.Series(index=mi_prior, data=prior_policies.flatten())
            policy_prior_df = policy_prior_df.unstack(level='policy').reset_index()

            policy_like_df =  pd.Series(index=mi_like, data=agents[w].likelihood.flatten())
            policy_like_df = policy_like_df.unstack(level='policy').reset_index()
            
            policy_entropy_df = pd.Series(index=mi,\
                data=policy_entropy[w].flatten()).reset_index().rename(columns = {0:'policy_entropy'})
            
            policy_surprise_df = pd.Series(index=mi,\
                data=policy_surprise[w].flatten()).reset_index().rename(columns = {0:'context_obs_surprise'})
            outcome_surprise_df = pd.Series(index=mi,\
                data=outcome_surprise[w].flatten()).reset_index().rename(columns = {0:'outcome_surprise'})
                
            # break
            df = pd.Series(index=mi, data=posterior_context[w].flatten())
            df = df.reset_index().rename(columns = {0:'probability'})
            df['context_obs_surprise'] = policy_surprise_df['context_obs_surprise']
            df['outcome_surprise'] = outcome_surprise_df['outcome_surprise']
            df['policy_entropy'] = policy_entropy_df['policy_entropy']
            # df['policy_entropy'] = policy_entropy_df['policy_entropy']
            df['learn_rew'] = np.repeat(meta['learn_rew'], factor)
            df['switch_cues'] = np.repeat(meta['switch_cues'], factor)
            df['contingency_degradation'] = np.repeat(meta['contingency_degradation'], factor)
            df['trials_per_block'] = np.repeat(meta['trials_per_block'], factor)
            df['degradation_blocks'] = np.repeat(meta['degradation_blocks'], factor)
            df['training_blocks'] = np.repeat(meta['training_blocks'], factor)
            df['context_cues'] = np.repeat(worlds[0].environment.context_cues, nc*t)
            df['true_optimal'] = np.repeat(meta['optimal_sequence'], nc*t)
            df['q'] = np.repeat(meta['context_trans_prob'], factor)
            df['p'] = np.repeat(meta['cue_ambiguity'], factor)
            df['h'] = np.repeat(meta['h'], factor)
            df['run'] = np.repeat(w,factor)
            df['trial_type'] = np.repeat(meta['trial_type'], nc*t)
            df['trial'] = np.repeat(np.arange(ntrials), nc*t)
            df['agent'] = np.repeat(w+f*nw,factor)
            df = df.join(policy_post_df.iloc[:,-8:])
            dfs[w] = df
            break
        overall_df[f] = pd.concat(dfs)
    data_animation = pd.concat(overall_df)
    # data_animation.to_excel('data_animation.xlsx')
    return data_animation
# data_animation.to_csv('data_animation.csv')


def load_df_animation_pol(names,data_folder='temp'):

    path = os.path.join(os.getcwd(),data_folder)
    #     names = os.listdir(path)

    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)


    overall_df = [None for _ in range(len(names))]
    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()

        with open(fname, 'r') as infile:
            data = json.load(infile)

        worlds = pickle.decode(data)
        meta = worlds[-1]
        nw = len(worlds[:-1])
        agents = [w.agent for w in worlds[:-1]]
        posterior_context = [agent.posterior_context for agent in agents]
        ntrials, t, nc = posterior_context[0].shape
        policy_entropy = [agent.policy_entropy for agent in agents]
        npi = agents[0].posterior_policies.shape[2]
        taus = np.arange(ntrials)
        ts = np.arange(t)
        cs = np.arange(nc)
        pis = np.arange(npi)

        mi = pd.MultiIndex.from_product([taus, ts, pis, cs], names=['trial', 't','policy', 'context'])
        dfs = [None for _ in range(nw)]
        factor = ntrials*nc*t*npi

        for w in range(nw):
            df =  pd.Series(index=mi, data=agents[w].posterior_policies.flatten())
            df = df.reset_index().rename(columns = {0:'policy_post'})

            prior_policies = np.tile(agents[w].prior_policies[:,np.newaxis,:,:], (1,t,1,1))
            policy_prior_df =  pd.Series(index=mi, data=prior_policies.flatten())
            policy_prior_df = policy_prior_df.reset_index().rename(columns = {0:'policy_prior'})

            policy_like_df =  pd.Series(index=mi, data=agents[w].likelihood.flatten())
            policy_like_df = policy_like_df.reset_index().rename(columns = {0:'policy_likelihood'})
            df['policy_prior'] = policy_prior_df['policy_prior']
            df['policy_like'] = policy_like_df['policy_likelihood']
            post_context = np.tile(posterior_context[w][:,:,np.newaxis,:], (1,1,npi,1))
            # print(np.all(post_context[:,:,0,:] == post_context[:,:,3,:]))

            post_context_df = pd.Series(index=mi, data=post_context.flatten())
            post_context_df = post_context_df.reset_index().rename(columns = {0:'post_context'})
            df['post_context'] = post_context_df['post_context']
            df['learn_rew'] = np.repeat(meta['learn_rew'], factor)
            df['switch_cues'] = np.repeat(meta['switch_cues'], factor)
            df['contingency_degradation'] = np.repeat(meta['contingency_degradation'], factor)
            df['trials_per_block'] = np.repeat(meta['trials_per_block'], factor)
            df['degradation_blocks'] = np.repeat(meta['degradation_blocks'], factor)
            df['training_blocks'] = np.repeat(meta['training_blocks'], factor)
            df['context_cues'] = np.repeat(worlds[0].environment.context_cues, nc*t*npi)
            df['true_optimal'] = np.repeat(meta['optimal_sequence'], nc*t*npi)
            df['q'] = np.repeat(meta['context_trans_prob'], factor)
            df['p'] = np.repeat(meta['cue_ambiguity'], factor)
            df['h'] = np.repeat(meta['h'], factor)
            df['run'] = np.repeat(w,factor)
            df['trial_type'] = np.repeat(meta['trial_type'], nc*t*npi)
            df['trial'] = np.repeat(np.arange(ntrials), nc*t*npi)
            df['agent'] = np.repeat(w+f*nw,factor)
            dfs[w] = df

        overall_df[f] = pd.concat(dfs)
    data_animation = pd.concat(overall_df)
    # data_animation.to_excel('data_animation.xlsx')
    return data_animation


def load_df_reward_dkl(names,data_folder='temp',nc=4):

    path = os.path.join(os.getcwd(),data_folder)
    #     names = os.listdir(path)

    for fi, f in enumerate(names):
        names[fi] = os.path.join(path,f)

    dfs = [None]*len(names)

    planet_reward_probs = np.array([[0.95, 0   , 0   ],
                            [0.05, 0.95, 0.05],
                            [0,    0.05, 0.95]])  

    planet_reward_probs_switched = np.array([[0   , 0    , 0.95],
                                            [0.05, 0.95 , 0.05],
                                            [0.95, 0.05 , 0.0]])
    
    planet_reward_probs = np.tile(planet_reward_probs[:,:,np.newaxis], (1,1,nc))
    planet_reward_probs_switched = np.tile(planet_reward_probs_switched[:,:,np.newaxis], (1,1,nc))

    overall_df = [None for _ in range(len(names))]

    for f,fname in enumerate(names):
        jsonpickle_numpy.register_handlers()
        with open(fname, 'r') as infile:
            data = json.load(infile)
                         
        worlds = pickle.decode(data)
        meta = worlds[-1]
        agents = [w.agent for w in worlds[:-1]]
        perception = [w.agent.perception for w in worlds[:-1]]
        reward_probs = [agent.posterior_dirichlet_rew for agent in agents]
        prior_rewards = worlds[0].agent.perception.prior_rewards

        nt = worlds[0].T
        npl = perception[0].npl
        nr = worlds[0].agent.nr
        nc = perception[0].nc
        nw = len(worlds[:-1])
        ntrials = meta['trials']
        # learn_rew = np.repeat(meta['learn_rew'], ntrials*nw*nt)
        # switch_cues = np.repeat(meta['switch_cues'], ntrials*nw*nt)
        # contingency_degradation = np.repeat(meta['contingency_degradation'], ntrials*nw*nt)
        # ntrials_df = np.repeat(meta['trials_per_block'], ntrials*nw*nt)
        # ndb = np.repeat(meta['degradation_blocks'], ntrials*nw*nt)
        # ntb = np.repeat(meta['training_blocks'], ntrials*nw*nt)
        post_dir_rewards = [a.posterior_dirichlet_rew for a in agents]
        post_dir_rewards = [post[:,1:,:,:] for post in post_dir_rewards]
        # entropy_rewards = np.zeros([nw*ntrials*nt,nc])
        # extinguished = np.zeros(ntrials*nw*nt, dtype='int32')
        # extinguished[:] = int(extinguish == True)
        
        # define true distribution reward
        # if f == 0:
        tpb = meta['trials_per_block']
        db = meta['degradation_blocks']
        tb = meta['training_blocks']
        p= np.tile(planet_reward_probs[np.newaxis,np.newaxis,:,:,:], (ntrials, nt, 1,1,1))
        p[tb*tpb:(tb+db)*tpb,:,:,:,:] = \
            np.tile(planet_reward_probs_switched[np.newaxis,np.newaxis,:,:,:],
                    ((db + tb)*tpb - tb*tpb, nt, 1,1,1))
        p[p == 0] = 10**(-300)
        # else:
        #     pass

        factor = ntrials*nt*nc
        taus = np.arange(ntrials)
        ts = np.arange(nt)
        # npls = np.char.add(np.asarray(['pl_' for _ in range(npl)]), np.asarray([str(i) for i in range(npl)]))
        npls = np.arange(npl)
        nrs = np.arange(nr)
        cs = np.arange(nc)
        mi = pd.MultiIndex.from_product([taus, ts, npls, cs],
                names=['trial', 't', 'planet', 'context'])
        # dfs_dkl = [None for _ in range(nw)]
        # factor = ntrials*nt*nr*npl*nc

        dkl_df = [None for _ in range(nw)]
        for w in range(nw):
            q = reward_probs[w]
            e = (db+tb)*tpb
            q[e:,:,:,:,:] = np.tile(q[e-1,:,:,:,:], (2*tpb,1,1,1,1))
            q[q == 0] = 10**(-300)
            norm = 1/(q.sum(axis=2))
            q = np.einsum('etrpc, etpc -> etrpc', q, norm)
            dkl = (q*np.log(q/p)).sum(axis=2)
            df =  pd.Series(index=mi, data=dkl.flatten())
            df = df.unstack(level = 'planet')
            df = df.reset_index().rename(columns = {0:'p0_dkl', 1:'p1_dkl', 2:'p2_dkl'})
            df['avg_dkl'] = (df['p0_dkl'] + df['p1_dkl'] + df['p2_dkl'])/3
            df['learn_rew'] = np.repeat(meta['learn_rew'], factor)
            df['switch_cues'] = np.repeat(meta['switch_cues'], factor)
            df['contingency_degradation'] = np.repeat(meta['contingency_degradation'], factor)
            df['trials_per_block'] = np.repeat(meta['trials_per_block'], factor)
            df['degradation_blocks'] = np.repeat(meta['degradation_blocks'], factor)
            df['training_blocks'] = np.repeat(meta['training_blocks'], factor)
            df['context_cues'] = np.repeat(worlds[0].environment.context_cues, nc*nt)
            df['true_optimal'] = np.repeat(meta['optimal_sequence'], nc*nt)
            df['q'] = np.repeat(meta['context_trans_prob'], factor)
            df['p'] = np.repeat(meta['cue_ambiguity'], factor)
            df['dec_temp'] = np.repeat(worlds[0].dec_temp,factor)
            df['h'] = np.repeat(meta['h'], factor)
            df['run'] = np.repeat(w,factor)
            df['trial_type'] = np.repeat(meta['trial_type'], nc*nt)
            df['trial'] = np.repeat(np.arange(ntrials), nc*nt)
            df['agent'] = np.repeat(w+f*nw,factor)
            df['utility_0'] = np.repeat(prior_rewards[0], factor)
            df['utility_1'] = np.repeat(prior_rewards[1], factor)
            df['utility_2'] = np.repeat(prior_rewards[2], factor)
            df['r_lambda'] = np.repeat(worlds[0].agent.perception.r_lambda, factor)
            dkl_df[w] = df
        #     break
        # break
        overall_df[f] = pd.concat(dkl_df)
    data = pd.concat(overall_df)
    return data



# %%
nc = 4
extinguish = True



hs =  [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
# h = [40]
cue_ambiguity = [0.65]#,0.75,0.8,0.85,0.9]                       
context_trans_prob = [0.8]#,0.85,0.9]
cue_switch = [False]
reward_naive = [True]
training_blocks = [4]
degradation_blocks=[2]
degradation = [True]
trials_per_block=[70]
dec_temps = [1,2]
rews = [0]
utility = [[1, 9, 90]]#, [5,25,70],[1,1,98],[1, 9, 90]]
conf = ['shuffled','shuffled_and_blocked']


# conf = ['shuffled', 'shuffled_and_blocked']

arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,\
        training_blocks, degradation_blocks, trials_per_block,dec_temps,rews, conf]

lst = []
for i in product(*arrays):
    lst.append(list(i))

for l in lst:
    names_arrays = [[l[0]], [l[1]], [l[2]], [l[3]], [l[4]],hs,\
            [l[5]], [l[6]], [l[7]],[l[8]],[l[9]], utility, [l[10]]] 
    print(names_arrays)
    data_folder = 'temp/'+l[10]
    names = load_file_names(names_arrays)
    df = load_df(names, data_folder=data_folder,extinguish=extinguish)
    df_dkl = load_df_reward_dkl(names, data_folder=data_folder)
    df.head()


    switch = l[0]
    contingency_degr = l[1]
    reward_naive = l[2]

    h = 200
    q = l[3]
    p = l[4]
    t = 3
    trials_per_block = l[7]
    training_blocks = l[5]
    db = l[6]
    degradation_blocks = [l[6]]
    dec_temp = l[8]
    cue = 0
    one_run = True
    rew = l[9]
    queries =  ['p==' + str(p)]

    for util in utility:
        
        util = [u/100 for u in util]

        strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
                        '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
                        '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
        vals = np.array([switch, contingency_degr, reward_naive, q, h,\
                        training_blocks, db, trials_per_block, dec_temp, util[0], util[1], util[2],rew], dtype='str')
        whole_query = np.char.join('', np.char.add(strs, vals))
        base_query = ' '.join(whole_query.tolist())
        if one_run == True:
            base_query += ' & run == 0'

        base_df = df.query(base_query)

        p = queries[0]
        # print(base_query)
        sns.set_style("whitegrid")
        fig = context_plot_cue_dependent(p,util = str(util), save=True)

        fname = 'figs/'+ l[-1] + '_'
        fname += '_'.join(['multiple_context','switch', str(int(l[0])), 'degr', str(int(l[1])),\
              'learn', str(int(l[2])), 'q', str(l[3]),\
              'p', str(l[4]),'dec', str(l[8]),'util', '-'.join([str(u) for u in util])]) + '.png'
        
        fig.savefig(fname, dpi=300)
        plt.close()

    if len(utility) == 1:
        utlty = [utility[0], utility[0]]
    else:
        utlty = utility

    sns.set_style('whitegrid')
    # for p in cue_ambiguity:
    fig, axes = plt.subplots(nrows=2, ncols=len(utlty), figsize = (len(utlty)*5,7))
    fig.suptitle('p=' + str(p))
    plt.tight_layout()


    # for di, db in enumerate(degradation_blocks):
    for ui, util in enumerate(utlty):
        db = degradation_blocks[0]
        util = [u/100 for u in util]
        # print(util)
        strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=', '& p==',\
                        '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
                        '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
                        
        vals = np.array([switch, contingency_degr, reward_naive, q, h, p, \
                        training_blocks, db, trials_per_block, dec_temp, util[0], util[1], util[2],rew], dtype='str')
        whole_query = np.char.join('', np.char.add(strs, vals))

        # strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
        #                 '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp =='],dtype='str')
        # vals = np.array([switch, contingency_degr, reward_naive, q, h,\
        #                 training_blocks, db, trials_per_block, dec_temp], dtype='str')
        # whole_query = np.char.join('', np.char.add(strs, vals))
        base_query = ' '.join(whole_query.tolist())
        # print(base_query)
        if one_run == True:
            base_query += ' & run == 0'

        base_df = df.query(base_query)
        base_df = base_df.astype({'h': 'category'})
        plot_df = base_df.query('trial_type == 1  & t==0 & degradation_blocks == ' + str(db))
        grouped = plot_df.groupby(by=['agent', 'run','h','cue'])
        plot_df['policy_optimality_subset'] = grouped['chose_optimal'].transform('cumsum')
        plot_df['offset'] = grouped['ith_cue_trial'].transform('min')
        plot_df['policy_optimality_subset'] = plot_df['policy_optimality_subset'] / (plot_df['ith_cue_trial'] - plot_df['offset']+1)



        for cue in [0,1]:
            sns.lineplot(ax = axes[cue,ui], data=plot_df.query('cue ==' + str(cue)),\
                        x = 'trial',y='policy_optimality_subset', hue='h', legend=False,\
                        palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size))
            axes[cue,ui].set_title('cue=' + str(cue) + ', ' + str(util))


        ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
        cols = [[1,1,1], [0,0,0],[1,1,1]] 
        for ax in axes.flatten():
            ax.set_ylim([0,1])
            for i, row in ranges.iterrows():
                ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)

        ax.legend(bbox_to_anchor=(-2,-0.3), loc='upper left', borderaxespad=0,title='h',ncol=np.unique(plot_df['h'].size))


            # fig.suptitle(ttls[f] + ' and cue_degradation: ' + str(cue_degradation[ci]) , fontsize=15, y=1.08)   
        # plot_df.to_excel(str(db) + '.xlsx')
        
    fname = 'figs/'+ l[-1] + '_'
    fname += '_'.join(['multiple_deval','switch', str(int(l[0])), 'degr', str(int(l[1])),\
            'learn', str(int(l[2])), 'q', str(l[3]),\
            'p', str(l[4]),'dec', str(l[8]),'util', '-'.join([str(u) for u in util])]) + '.png'

    fig.savefig(fname, dpi=300)
    plt.close()




    for ui, util in enumerate(utility):
        util = [u/100 for u in util]
        # print(db)
        strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=', '& p==', \
                        '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
                        '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
        vals = np.array([switch, contingency_degr, reward_naive, q, h, p,\
                        training_blocks, db, trials_per_block, dec_temp, util[0], util[1], util[2], rew], dtype='str')
        whole_query = np.char.join('', np.char.add(strs, vals))
        base_query = ' '.join(whole_query.tolist())

        base_df_dkl = df_dkl.query(base_query)

        plot_df = base_df_dkl.query('t==3' + ' & degradation_blocks ==' + str(db))
        # print(plot_df.shape)
        fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20,3))
        for cont in range(4):

            # print('context_cues == ' + str(cue) + ' & context== ' + str(cont))
            if one_run == True:
                quer = 'context== ' + str(cont) + ' & run == 0'
            else:
                quer = 'context== ' + str(cont)
            if cont == 3:
                sns.lineplot(ax=axes[cont], data=plot_df.query(quer),
                            x='trial', y='avg_dkl', hue='h', legend=True, \
                            palette=sns.color_palette('Reds_r',n_colors=np.unique(plot_df['h']).size))
                # print(np.unique(plot_df['h']))
                axes[cont].legend(ncol = np.unique(plot_df['h']).size, bbox_to_anchor=(-3.3, -0.3), loc='upper left',\
                    borderaxespad=0,title='h')
            else:
                
                sns.lineplot(ax=axes[cont], data=plot_df.query(quer),
                            x='trial', y='avg_dkl', hue='h',legend=False,\
                                palette=sns.color_palette('Reds_r',n_colors=np.unique(plot_df['h']).size))
            axes[cont].set_title('context: ' + str(cont))

        ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
        cols = [[1,1,1], [0,0,0],[1,1,1]] 
        for ax in axes.flatten():    
            for i, row in ranges.iterrows():
                ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
        plt.suptitle(str(util))
    # plt.figure()

        fname = 'figs/'+ l[-1] + '_'

        fname += '_'.join(['multiple_dkl','switch', str(int(l[0])), 'degr', str(int(l[1])),\
                'learn', str(int(l[2])), 'q', str(l[3]),\
                'p', str(l[4]),'dec', str(l[8]),'util', '-'.join([str(u) for u in util])]) + '.png'

        fig.savefig(fname, dpi=300)
        plt.close()
        

























#%%



# arrays = [cue_switch, degradation, reward_naive, context_trans_prob, cue_ambiguity,h,\
#         training_blocks, degradation_blocks, trials_per_block,dec_temps,rews, utility, conf]

# # pooled(arrays)
# data_folder = 'temp/'+conf[0]
# names = load_file_names(arrays)
# df = load_df(names, data_folder=data_folder,extinguish=extinguish)
# df_dkl = load_df_reward_dkl(names, data_folder=data_folder)
# df.head()




# # ### Context optimality and choice optimality plots


# switch = False
# contingency_degr = True
# reward_naive = True

# h = 200
# q = context_trans_prob[-1]
# p = cue_ambiguity[2]
# t = 3
# trials_per_block = 70
# training_blocks = 4
# db = degradation_blocks[0]
# degradation_blocks = [6]
# dec_temp = dec_temps[1]
# cue = 0
# one_run = True
# rew = rews[0]
# queries =  ['p==' + str(p)]

# # for db in degradation_blocks:
# for util in utility:
    
#     util = [u/100 for u in util]
#     # print(cue_ambiguity[ind], q,util)
#     strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
#                     '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
#                     '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
#     vals = np.array([switch, contingency_degr, reward_naive, q, h,\
#                     training_blocks, db, trials_per_block, dec_temp, util[0], util[1], util[2],rew], dtype='str')
#     whole_query = np.char.join('', np.char.add(strs, vals))
#     base_query = ' '.join(whole_query.tolist())
#     if one_run == True:
#         base_query += ' & run == 0'

#     base_df = df.query(base_query)
    
#     # print(base_df.size) # base_df_dkl = df_dkl.query(base_query)
#     # queries = ['p==0.95','p==0.7']
#     p = queries[0]
#     # print(base_query)
#     sns.set_style("whitegrid")
#     context_plot_cue_dependent(p,util = str(util))



# print(base_query)

# ################################## optimality reward


# if len(utility) == 1:
#     utlty = [utility[0], utility[0]]
# else:
#     utlty = utility
# # print(utlty)

# sns.set_style('whitegrid')
# # for p in cue_ambiguity:
# fig, axes = plt.subplots(nrows=2, ncols=len(utlty), figsize = (len(utlty)*5,7))
# fig.suptitle('p=' + str(p))
# plt.tight_layout()


# # for di, db in enumerate(degradation_blocks):
# for ui, util in enumerate(utlty):
#     db = degradation_blocks[0]
#     util = [u/100 for u in util]
#     # print(util)
#     strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=', '& p==',\
#                     '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
#                     '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
                    
#     vals = np.array([switch, contingency_degr, reward_naive, q, h, p, \
#                     training_blocks, db, trials_per_block, dec_temp, util[0], util[1], util[2],rew], dtype='str')
#     whole_query = np.char.join('', np.char.add(strs, vals))

#     # strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
#     #                 '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp =='],dtype='str')
#     # vals = np.array([switch, contingency_degr, reward_naive, q, h,\
#     #                 training_blocks, db, trials_per_block, dec_temp], dtype='str')
#     # whole_query = np.char.join('', np.char.add(strs, vals))
#     base_query = ' '.join(whole_query.tolist())
#     # print(base_query)
#     if one_run == True:
#         base_query += ' & run == 0'

#     base_df = df.query(base_query)
#     base_df = base_df.astype({'h': 'category'})
#     plot_df = base_df.query('trial_type == 1  & t==0 & degradation_blocks == ' + str(db))
#     grouped = plot_df.groupby(by=['agent', 'run','h','cue'])
#     plot_df['policy_optimality_subset'] = grouped['chose_optimal'].transform('cumsum')
#     plot_df['offset'] = grouped['ith_cue_trial'].transform('min')
#     plot_df['policy_optimality_subset'] = plot_df['policy_optimality_subset'] / (plot_df['ith_cue_trial'] - plot_df['offset']+1)



#     for cue in [0,1]:
#         sns.lineplot(ax = axes[cue,ui], data=plot_df.query('cue ==' + str(cue)),\
#                     x = 'trial',y='policy_optimality_subset', hue='h', legend=False,\
#                     palette=sns.color_palette('Blues_r',n_colors=np.unique(plot_df['h']).size))
#         axes[cue,ui].set_title('cue=' + str(cue) + ', ' + str(util))


#     ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
#     cols = [[1,1,1], [0,0,0],[1,1,1]] 
#     for ax in axes.flatten():
#         ax.set_ylim([0,1])
#         for i, row in ranges.iterrows():
#             ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)

#     ax.legend(bbox_to_anchor=(-2,-0.3), loc='upper left', borderaxespad=0,title='h',ncol=np.unique(plot_df['h'].size))


#         # fig.suptitle(ttls[f] + ' and cue_degradation: ' + str(cue_degradation[ci]) , fontsize=15, y=1.08)   
#     # plot_df.to_excel(str(db) + '.xlsx')
    


# ################################## DKL

# one_run = True
# # util = utility[-1]
# # util = [u/100 for u in util]

# # strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=',\
# #                 '& training_blocks==', '& degradation_blocks<=10', '& trials_per_block==', '& dec_temp =='],dtype='str')

# # vals = np.array([switch, contingency_degr, reward_naive, q, h,\
# #                 training_blocks, degradation_blocks, trials_per_block, dec_temp], dtype='str')


# # for di, db in enumerate(degradation_blocks):

# for ui, util in enumerate(utility):
#     util = [u/100 for u in util]
#     # print(db)
#     strs = np.array(['switch_cues==', '& contingency_degradation==', '& learn_rew==', '& q==', '& h<=', '& p==', \
#                     '& training_blocks==', '& degradation_blocks==', '& trials_per_block==', '& dec_temp ==',\
#                     '& utility_0==', '& utility_1==', '& utility_2==', '& r_lambda=='],dtype='str')
#     vals = np.array([switch, contingency_degr, reward_naive, q, h, p,\
#                     training_blocks, db, trials_per_block, dec_temp, util[0], util[1], util[2], rew], dtype='str')
#     whole_query = np.char.join('', np.char.add(strs, vals))
#     base_query = ' '.join(whole_query.tolist())

#     base_df_dkl = df_dkl.query(base_query)

#     plot_df = base_df_dkl.query('t==3' + ' & degradation_blocks ==' + str(db))
#     # print(plot_df.shape)
#     fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20,3))
#     for cont in range(4):

#         # print('context_cues == ' + str(cue) + ' & context== ' + str(cont))
#         if one_run == True:
#             quer = 'context== ' + str(cont) + ' & run == 0'
#         else:
#             quer = 'context== ' + str(cont)
#         if cont == 3:
#             sns.lineplot(ax=axes[cont], data=plot_df.query(quer),
#                         x='trial', y='avg_dkl', hue='h', legend=True, \
#                         palette=sns.color_palette('Reds_r',n_colors=np.unique(plot_df['h']).size))
#             # print(np.unique(plot_df['h']))
#             axes[cont].legend(ncol = np.unique(plot_df['h']).size, bbox_to_anchor=(-3.3, -0.3), loc='upper left',\
#                 borderaxespad=0,title='h')
#         else:
            
#             sns.lineplot(ax=axes[cont], data=plot_df.query(quer),
#                         x='trial', y='avg_dkl', hue='h',legend=False,\
#                             palette=sns.color_palette('Reds_r',n_colors=np.unique(plot_df['h']).size))
#         axes[cont].set_title('context: ' + str(cont))

#     ranges = plot_df.groupby('trial_type')['trial'].agg(['min', 'max'])
#     cols = [[1,1,1], [0,0,0],[1,1,1]] 
#     for ax in axes.flatten():    
#         for i, row in ranges.iterrows():
#             ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.05)
#     plt.suptitle(str(util))
# plt.figure()

# # %%
