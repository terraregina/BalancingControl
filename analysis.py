
#%% 
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

file = open("run_object.txt",'rb')
run = pickle.load(file)

file.close()


trials = run.trials
agent = run.agent
perception = agent.perception
environment= run.environment

observations = run.observations
true_optimal = environment.true_optimal
context_cues = environment.context_cues
trial_type = environment.trial_type
policies = run.agent.policies
actions = run.actions[:,:3] 
executed_policy = np.zeros(trials)

# 

agent.posterior_context
agent.posterior_rewards
agent.posterior_policies
agent.likelihood
for pi, p in enumerate(policies):
    inds = np.where( (actions[:,0] == p[0]) & (actions[:,1] == p[1]) & (actions[:,2] == p[2]) )[0]
    executed_policy[inds] = pi
reward = run.rewards


data = pd.DataFrame({'executed': executed_policy,
                     'optimal': true_optimal,
                     'trial': np.arange(true_optimal.size),
                     'trial_type': trial_type})

data['chose_optimal'] = data.executed == data.optimal
data['optimality'] = np.cumsum(data['chose_optimal'])/(data['trial']+1)
fig = plt.figure()
plt.subplot(1,2,1)
ax = sns.scatterplot(data=data[['executed','chose_optimal']])
cols = [[0,1,1], [1,0,0],[0,1,1]] 
ranges = data.groupby('trial_type')['trial'].agg(['min', 'max'])
for i, row in ranges.iterrows():
    ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

plt.subplot(1,2,2)
ax = sns.lineplot(data=data, x=data['trial'], y=data['optimality'])


for i, row in ranges.iterrows():
    ax.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

fig.figure.savefig('1.png', dpi=300) 
# plt.close()

#%%
t = -1

posterior_context = agent.posterior_context
data = pd.DataFrame({'posterior_h3b': posterior_context[:t,3,0],
                     'posterior_h6o': posterior_context[:t,3,1],
                     'posterior_h6b': posterior_context[:t,3,2],
                     'posterior_h3o': posterior_context[:t,3,3],
                     'context_cue': context_cues[:t],    
                     'trial_type': trial_type[:t],
                     'trial': np.arange(trial_type[:t].size)})


cols = [[0,1,1], [1,0,0],[0,1,1]] 
fig2 = sns.lineplot(data=data.iloc[:,:4])
ranges = data.groupby('trial_type')['trial'].agg(['min', 'max'])
for i, row in ranges.iterrows():
    fig2.axvspan(xmin=row['min'], xmax=row['max'], facecolor=cols[i], alpha=0.1)

# fig2 = sns.scatterplot(data=data.iloc[:,4])
# fig2.figure.savefig('2.png',dpi=300)


#%%

posterior_rewards = agent.posterior_dirichlet_rew
print(posterior_rewards.shape)

