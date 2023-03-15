# 10k each type as KNe and cont
# not adding SBO due absent _all
import pandas as pd
import numpy as np
import random

import utils

band_list=utils.get_band_info('ZTF_full')
times = np.arange(0,7,0.1)

def get_lcs(types):
  lc_set = pd.DataFrame()
  for type in types:
    lcs = pd.read_csv('data/'+type+'.csv')
    lcs = lcs.replace(np.inf, np.nan).dropna()
    bad_sims = []
    for sim,lc in lcs.groupby('sim'):
      lc['passband'] = lc['passband'].map({'g':1, 'r':2, 'i':3})
      GP_lc = pd.DataFrame(data=np.vstack((np.repeat(times,len(band_list)),
                                            np.tile(band_list,len(times))
                                           )).T,columns=['mjd','passband'])
      try:
        GP_lc = utils.GP_free((lc['mjd']-lc['tc']).values,
                          lc['mag'].values,
                          lc['mag_err'].values,
                          lc['passband'].values,
                          GP_lc,
                          1.)
      except ValueError:
        bad_sims.append(sim)
    lcs = lcs[~lcs['sim'].isin(bad_sims)]
    print(type+' n valid sims '+str(lcs.groupby('sim').ngroups)) #should be > 10000
    lcs = lcs[lcs['sim'].isin(random.sample(lcs['sim'].unique().tolist(), 10000))]
    if type in ['BNS', 'NSBH', 'GRB', 'SBO']: # converting to type_sim-num_ToO_rand
      lcs['sim'] = lcs['sim'].apply(lambda x: x.split('/')[1]+'_'+x.split('/')[2])
    else: #becomes SNe_sim-num_ToO
      lcs['sim'] = lcs['sim'].apply(lambda x: type+'_'+x)
    lc_set = pd.concat([lc_set, lcs])
  return lc_set

types = ['BNS', 'NSBH']
KN_lc_set = get_lcs(types)
types = list(utils.types.keys()) + ['GRB', 'SBO']
contaminant_lc_set = get_lcs(types)

#recompute below for _all files
def get_lcs(types, sims):
  lc_set = pd.DataFrame()
  for type in types:
    lcs = pd.read_csv('data/'+type+'_all.csv')
    lcs = lcs.replace(np.inf, np.nan).dropna()
    if type in ['BNS', 'NSBH', 'GRB']: # converting to type_sim-num_rand
      lcs['sim'] = lcs['sim'].apply(lambda x: x.split('/')[1]+'_'+x.split('/')[2].split('_all')[0])
    else: #becomes SNe_sim-num_ToO
      lcs['sim'] = lcs['sim'].apply(lambda x: type+'_'+x)
    lcs = lcs[lcs['sim'].isin(sims)]
    lc_set = pd.concat([lc_set, lcs])
  return lc_set

sims = []
for sim in KN_lc_set.groupby('sim').groups.keys():
  ids = sim.split('_')
  sims.append(ids[0] + '_' + ids[1] + '_' + ids[3]) #both ToOs have same base signal, not sure re effect of rand seed on LC itself
sims = [*set(sims)] #removes duplicates
types = ['BNS', 'NSBH']
KN_lc_set_all = get_lcs(types, sims)

if set(sims) != set(KN_lc_set_all.groupby('sim').groups.keys()):
  print('not all matches found')

sims = []
for sim in contaminant_lc_set.groupby('sim').groups.keys():
  if 'GRB' in sim:
    ids = sim.split('_')
    sims.append(ids[0] + '_' + ids[1] + '_' + ids[3]) #both ToOs have same base signal, not sure re effect of rand seed on LC itself
sims = [*set(sims)] #removes duplicates
types = ['GRB']
contaminant_lc_set_GRB = get_lcs(types, sims)

if set(sims) != set(contaminant_lc_set_GRB.groupby('sim').groups.keys()):
  print('not all matches found')

sims = []
for sim in contaminant_lc_set.groupby('sim').groups.keys():
  if sim.startswith(tuple(utils.types.keys())):
    sims.append(sim)
contaminant_lc_set_SNe = get_lcs(list(utils.types.keys()), sims)

if set(sims) != set(contaminant_lc_set_SNe.groupby('sim').groups.keys()):
  print('not all matches found')

contaminant_lc_set_all = pd.concat([contaminant_lc_set_GRB,
                                    contaminant_lc_set_SNe[['mjd','passband','mag','sim','luminosity_distance']]
                                  ])

KN_lc_set.to_csv('data/KNe.csv', index=False)
contaminant_lc_set.to_csv('data/contaminant.csv', index=False)

KN_lc_set_all.to_csv('data/KNe_all.csv', index=False)
contaminant_lc_set_all.to_csv('data/contaminant_all.csv', index=False)

from sklearn.model_selection import train_test_split

#not equal types of each subtype
KN_train_sims, KN_test_sims = train_test_split(KN_lc_set['sim'].unique(), test_size=0.25)
KN_lc_train = KN_lc_set[KN_lc_set['sim'].isin(KN_train_sims)]
KN_lc_test = KN_lc_set[KN_lc_set['sim'].isin(KN_test_sims)]

contaminant_train_sims, contaminant_test_sims = train_test_split(contaminant_lc_set['sim'].unique(), test_size=0.25)
contaminant_lc_train = contaminant_lc_set[contaminant_lc_set['sim'].isin(contaminant_train_sims)]
contaminant_lc_test = contaminant_lc_set[contaminant_lc_set['sim'].isin(contaminant_test_sims)]

KN_lc_train.to_csv('data/KNe_train.csv', index=False)
KN_lc_test.to_csv('data/KNe_test.csv', index=False)
contaminant_lc_train.to_csv('data/contaminant_train.csv', index=False)
contaminant_lc_test.to_csv('data/contaminant_test.csv', index=False)
