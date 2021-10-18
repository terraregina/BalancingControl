#%%
 
from misc_sia import load_file
import os as os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np


ttls = ['npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_1.txt',
        'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_1.txt']

ttls = ['rdm_grid_policies_cont-1_prior-1_like_h1000_s0.001_wd1_b4_a4_1',
        'rdm_grid_policies_cont-1_prior-1_stand_h1000_s6.399999999999994e-05_wd0.1280639999999999_b1.0_a3.5_1']


for ttl in ttls:
    ww = load_file(os.getcwd() + '\\agent_sims\\desired_rt_1\\' + ttl)
    Q_policies = np.asarray(ww.agent.action_selection.drifts)[np.arange(0,800,4),:]
    priors = np.asarray(ww.agent.action_selection.priors)[np.arange(0,800,4),:]/3.5
    print(Q_policies[99:105,:].round(4))
    print(priors[99:105,:].round(4))
#%%
ttls = ['npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_1.txt',
        'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_1.txt']

ttls = ['rdm_grid_policies_cont-1_prior-1_like_h1000_s0.001_wd1_b4_a4_1',
        'rdm_grid_policies_cont-1_prior-1_stand_h1000_s6.399999999999994e-05_wd0.1280639999999999_b1.0_a3.5_1']


for ttl in ttls:

    ww = load_file(os.getcwd() + '\\agent_sims\\desired_rt_1\\' + ttl)
    Q_policies = np.asarray(ww.agent.action_selection.drifts)[np.arange(0,800,4),:]
    Q_policies = Q_policies[95:120]
    
    def init_animation():
        global line
        line, = ax.plot(x, np.zeros_like(x))
        ax.set_xlim(0, 81)
        ax.set_ylim(0,1)


    def animate(i):
        line.set_ydata(Q_policies[i,:])
        fig.suptitle(str(i+95))
        return line,

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0,81)


    ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init_animation, frames=25)
    ani.save(r'C:/Users/admin/Desktop/project/BalancingControl/' + ttl + '.gif', writer='imagemagick', fps=5)

#%%


# ttls = ['rdm_grid_policies_cont-1_prior-1_stand_h1000_s6.399999999999994e-05_wd0.1280639999999999_b1.0_a3.5_1','rdm_grid_policies_cont-1_prior-1_post_h1000_s0.001_wd1_b3_a4_1']
# ttls = ['rdm_grid_policies_cont-1_prior-1_stand_h1000_s6.399999999999994e-05_wd0.1280639999999999_b1.0_a3.5_0',
#         'rdm_grid_policies_cont-1_prior-1_stand_h1000_s6.399999999999994e-06_wd0.1280639999999999_b1.0_a3.5_0']
# for ttl in ttls:   
#     ww = load_file(os.getcwd() + '\\agent_sims\\desired_rt_1\\' + ttl)
#     # ww = load_file(os.getcwd() + '\\agent_sims\\rdm_grid_policies_cont-1_prior-1_post_h1000_s0.001_wd1_b3_a1')
#     ac_sel = ww.agent.action_selection
#     ac_sel.print_walks = True

#     Q_log = np.asarray(ww.agent.action_selection.drifts)[np.arange(0,800,4),:]

#     if ac_sel.prior_as_starting_point:
#         starts = np.asarray(ww.agent.action_selection.priors)[np.arange(0,800,4),:]


#     if ac_sel.over_actions:
#         ni=3
#     else:
#         ni=81


#     RT = []
#     for ind, tau in enumerate(range(99,130)):
#         Q = Q_log[tau,:]
#         decision_log = np.zeros([10000, ni])

#         if ac_sel.prior_as_starting_point:
#             decision_log[0,:] = starts[tau,:]
#         broken = False
#         i=0
        
#         bound_reached = np.zeros(ni, dtype = bool)
#         while(not np.any(bound_reached)):                        # if still no decision made
#             dW = np.random.normal(scale = ac_sel.s, size = ni)
#             decision_log[i+1,:] = decision_log[i,:] + (ac_sel.wd*Q)*0.01 + (Q>0)*dW
#             bound_reached = decision_log[i+1,:] >= ac_sel.b
#             i+=1

#             if i > 10000:
#                 action = -1
#                 broken = True
#                 break

#         crossed = bound_reached.nonzero()[0]                      # non-zero returns tuple by default
        
        
#         if (crossed.size > 1):
#             vals = decision_log[i,crossed]
#             selected = np.argmax(vals)
#             decision = crossed[selected]

#             # raise ValueError('Two integrators crossed decision boundary at the same time')
#             # print('Two integrators crossed boundary at the same time, choosing one at random')
#             decision = np.random.choice(crossed, p=np.ones(crossed.size)/crossed.size)
#         elif not broken:
#             decision = crossed[0]

#         RT.append(i)

#         plt.plot(np.arange(0, i+1), decision_log[:i+1,:])
#         plt.title('tau: ' +  str(tau) + ' RT:' + str(RT[ind]))
#         plt.savefig(ttl + '_' + str(tau) + '.png')
#         plt.close()
#     plt.close()
#     plt.plot(np.arange(99,130), RT,marker='o')
#     plt.savefig('RT_' + ttl + '.png')
