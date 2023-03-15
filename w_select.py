import os, glob
import numpy as np
import pandas as pd
import time
import random

import defs
import pythia

agent = 'Pythia'
epsilon = 0.
n_episodes = 100
data = defs.train_data('test')
outdir = 'outdir/test/w_select'

dest = 'outdir/n9'
alpha_list = [1e-2, 1e-3]
gamma_list = [0.9, 0.5, 0.1]
n = 3.
smooth = 50

if os.path.isfile(outdir+'/summary.csv'):
  summ = pd.read_csv(outdir+'/summary.csv')
else:
  summ = pd.DataFrame()

for i,alpha in enumerate(alpha_list):
  for j,gamma in enumerate(gamma_list):
    train_status = pd.read_csv(dest+'/train_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.csv')
    avg_score = train_status.groupby('k').mean()['R_tau'].rolling(window=smooth).mean()
    for idx in avg_score[avg_score > 2.8*(defs.horizon-1)/defs.N].index:
      if defs.epsilon_/idx**(1./n) >= 0.1: continue
      w_file = dest+'/train_linw_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'_'+str(idx)+'.npy'
      w_name = os.path.splitext(os.path.basename(w_file))[0]
      if os.path.exists(outdir+'/'+w_name):
        continue
      with open(w_file, 'rb') as f:
        weights = np.load(f)
      lcs = pd.DataFrame()
      behav = pd.DataFrame()
      score = pd.DataFrame()
      k = 0
      while k < n_episodes:
        t = time.time()
        #select episode
        KN = random.choice(data.KN_lc['sim'].unique().tolist()) #names
        contaminants = random.sample(data.contaminant_lc['sim'].unique().tolist(), defs.N - 1)
        KN_idx = random.randrange(defs.N)
        contaminant_idx = [i for i in range(defs.N) if i != KN_idx]
        random.shuffle(contaminant_idx)
        KN_lc_ = data.KN_lc[data.KN_lc['sim'] == KN]
        contaminant_lcs_ = data.contaminant_lc[data.contaminant_lc['sim'].isin(contaminants)]
        if KN_lc_['passband'].nunique() < defs.n_phot:
            continue
        if (contaminant_lcs_.groupby(['sim']).apply(lambda x: x['passband'].nunique()) < defs.n_phot).any():
            continue

        k += 1
        R_tau = 0
        KN_lc = KN_lc_.copy()
        contaminant_lcs = contaminant_lcs_.copy()
        for timestep in range(1, defs.horizon):
          state = pythia.State(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep)
          e_greedy_action = pythia.e_greedy(state, weights, epsilon)
          reward = pythia.get_reward(e_greedy_action, KN_idx)
          state_prime, KN_lc, contaminant_lcs, obs = pythia.next_state(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep+1, e_greedy_action)
          R_tau += reward
          info = [k, timestep, obs['sim'].item(), obs['passband'].item()]
          behav = pd.concat([behav, pd.DataFrame([info], columns = ['episode', 'timestep', 'event_chosen', 'obs_passband'])])
        score = pd.concat([score, pd.DataFrame([R_tau])])
        KN_lc['episode'] = k
        KN_lc['position'] = KN_idx
        contaminant_lcs['episode'] = k
        contaminant_lcs['position'] = contaminant_lcs['sim'].map(dict(zip(contaminants, contaminant_idx)))
        lcs = pd.concat([lcs, KN_lc, contaminant_lcs])
      os.makedirs(outdir+'/'+w_name)
      lcs.to_csv(outdir+'/'+w_name+'/lcs.csv', index=False)
      behav.to_csv(outdir+'/'+w_name+'/behav.csv', index=False)
      summ = pd.concat([summ, pd.DataFrame(
                                          [[w_name, score[0].mean(), score[0].std(), score[score[0] > 0].shape[0]/n_episodes]],
                                          columns = ['w_name', 'mean', 'stddev', 'frac > 0']
                                          )
                      ])
      summ.to_csv(outdir+'/summary.csv', index=False)
