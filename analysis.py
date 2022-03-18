
#%% 
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
#%%
file = open("run_object.txt",'rb')
run = pickle.load(file)

file.close()

#%%
trials = run.trials
agent = run.agent
perception = agent.perception
observations = run.observations
true_optimal = run.environment.true_optimal
context_cues = run.environment.context_cues
policies = run.agent.policies
actions = run.actions[:,:3] 
executed_policy = np.zeros(trials)

for pi, p in enumerate(policies):
    inds = np.where( (actions[:,0] == p[0]) & (actions[:,1] == p[1]) & (actions[:,2] == p[2]) )[0]
    executed_policy[inds] = pi
reward = run.rewards

#%%

data = pd.DataFrame({'executed': executed_policy, 'optimal': true_optimal})
data['chose_optimal'] = data.executed == data.optimal
sns.scatterplot(data=data)


# %%
posterior_context = agent.posterior_context
data = pd.DataFrame({'posterior_c1': posterior_context[:,0,0],\
                     'posterior_c2': posterior_context[:,1,0],\
                     'posterior_c3': posterior_context[:,2,0],\
                     'posterior_c4': posterior_context[:,3,0]})

sns.lineplot(data = data,dpi=300)
# %%
