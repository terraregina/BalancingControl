import numpy as np

h = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
# h = [1,100]
cue_ambiguity = [0.85]
context_trans_prob = [0.8]
cue_switch = [False]
reward_naive = [True]
training_blocks = [6]
degradation_blocks=[6]
degradation = [True]
trials_per_block=[42]
dec_temps = [1]
rews = [0]
rewards = [[-1,0,1]]
# for determinstic context update do a 100
dec_context = [1]


extinguish = True

na = 2                                           # number of unique possible actions
nc = 4                                           # number of contexts, planning and habit
nr = 3                                           # number of rewards
nr = len(rewards[0])
ns = 6                                           # number of unique travel locations
npl = nr
steps = 3                                        # numbe of decisions made in an episode
T = steps + 1                                    # episode length

print('NUMBER OF REWARDS', nr)

if npl == 3:
    planet_reward_probs = np.array([[0.95, 0   , 0   ],
                                    [0.05, 0.95, 0.05],
                                    [0,    0.05, 0.95]]).T    # npl x nr
    planet_reward_probs_switched = np.array([[0   , 0    , 0.95],
                                            [0.05, 0.95 , 0.05],
                                            [0.95, 0.05 , 0.0]]).T 


elif npl == 2:
    planet_reward_probs = np.array([[0.95,   0.05   ],
                                    [0.05,   0.95]]).T    # npl x nr

    planet_reward_probs_switched = np.array([[0.05,   0.95   ],
                                                [0.95,   0.05]]).T    # npl x nr

state_transition_matrix = np.zeros([ns,ns,na])
m = [1,2,3,4,5,0]
for r, row in enumerate(state_transition_matrix[:,:,0]):
    row[m[r]] = 1
j = np.array([5,4,5,6,2,2])-1

for r, row in enumerate(state_transition_matrix[:,:,1]):
    row[j[r]] = 1
state_transition_matrix = np.transpose(state_transition_matrix, axes= (1,0,2))
state_transition_matrix = np.repeat(state_transition_matrix[:,:,:,np.newaxis], repeats=nc, axis=3)

nc = 4



if nr == 3:
    utility = [[1, 9 , 90]]
else:
    utility = [[1,99]]
    
conf = ['shuffled_and_blocked']

hs = h
dec_temp_cont = dec_context
importing = True