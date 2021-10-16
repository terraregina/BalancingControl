#%%

import seaborn as sns
import matplotlib.pyplot as plt
from agent_simulations import make_ttl_from_params
from misc_sia import *
import os as os

pars_for_fig = ['npi_3_rdm_standard_b_1_wd_0.1280639999999999_s_6.399999999999994e-05_.txt', 'npi_81_rdm_standard_b_1.0_wd_0.1280639999999999_s_6.399999999999994e-05_a_3.5_.txt', 'npi_3_rdm_post_prior1_b_3_wd_1_s_0.0034_a_1_.txt', 'npi_81_rdm_post_prior1_b_3_wd_1_s_0.001_a_4_.txt', 'npi_3_rdm_like_prior1_b_7_wd_1_s_0.0034_a_2_.txt', 'npi_81_rdm_like_prior1_b_4_wd_1_s_0.001_a_4_.txt']	
pars = np.asarray(pars_for_fig)
pars = pars[np.newaxis,:].reshape([3,2])


regimes = ["'standard'", "'post_prior1'", "'like_prior1'"]
    
fig,axes = plt.subplots(3,2, figsize=(8,8))
#%%

for r, reg in enumerate(regimes):
    for ni, npi in enumerate([3,81]):

        data = load_file(os.getcwd() + '\\parameter_data\\' + pars[r,ni])

        rt_df = {
            'rts': data['RT'].ravel() ,
            'mode': tests.repeat(1000)
            }
        
        rt_df = pd.DataFrame(rt_df)
        
        sns.histplot(ax = axes[r, ni], \
                     data=rt_df,
                     x='rts', hue='mode', legend=False, bins=100)

axes[0,0].set_title('$n$ = 3')
axes[0,1].set_title('$n$ = 81')

axes[r, ni].legend(labels = ['habit','goal', 'agreement','conflict'], bbox_to_anchor=(-0.1, -0.6), loc='lower center', ncol=4)
plt.subplots_adjust(hspace=0.6)
axes = axes.flat
for n, ax in enumerate(axes):

    ax.text(-0.1, 1.2, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=15)


plt.savefig('example_parametrizations.png', dpi=300)
        
#%% CREATE TABLES
parameters = []
for file in pars_for_fig:
    parameters.append(extract_params(file))

load_fits_from_ttl(pars_for_fig):