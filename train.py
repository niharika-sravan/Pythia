import numpy as np
import random
import pandas as pd
import os, sys
import argparse
import warnings
import time
import joblib
import pickle

from scipy import stats

import utils
import pythia
import defs

dest = 'outdir/n9'
data = defs.train_data('train')

parser=argparse.ArgumentParser()
parser.add_argument("alpha", type=float)
parser.add_argument("gamma", type=float)
parser.add_argument("eps_decay", type=float)
parser.add_argument("initialize", type=str)
args=parser.parse_args()

alpha = args.alpha # learning rate parameter from Adam
gamma = args.gamma # hyperparameter to tune
n = args.eps_decay # epsilon decay rate; hyperparameter to tune
epsilon = defs.epsilon_
n_episodes = int((epsilon/0.02)**n) # should be large enough to be GLIE; set at 2%

if args.initialize == 'random':
  w = np.random.rand((defs.out_size + 1), 1) * 1e-3
  adam = utils.AdamOptim(alpha = alpha)
  k = 0
  train_status = pd.DataFrame()
elif args.initialize == 'resume': # in the output csv there may be lesser repeats of a given k
  try:
    with open(dest+'/train_linw_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.npy', 'rb') as f:
      w = np.load(f)
    with open(dest+'/adam_obj_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.pkl', 'rb') as f:
      adam = pickle.load(f)
    train_status = pd.read_csv(dest+'/train_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.csv')
    k = train_status['k'].iloc[-1]
  except FileNotFoundError as e:
    print('A run with these parameters does not exist. Use initialize random')
    raise e
else: # warm start with weights file
  with open(args.initialize, 'rb') as f:
    w = np.load(f)
  adam = utils.AdamOptim(alpha = alpha)
  k = 0
  train_status = pd.DataFrame()
delw = w

while k < n_episodes:
  #select episode
  KN = random.choice(data.KN_lc['sim'].unique().tolist()) #names
  contaminants = random.sample(data.contaminant_lc['sim'].unique().tolist(), defs.N - 1)
  KN_idx = random.randrange(defs.N)
  contaminant_idx = [i for i in range(defs.N) if i != KN_idx]
  random.shuffle(contaminant_idx)
  KN_lc_ = data.KN_lc[data.KN_lc['sim'] == KN]
  contaminant_lcs_ = data.contaminant_lc[data.contaminant_lc['sim'].isin(contaminants)]
  if KN_lc_['passband'].nunique() < defs.n_phot: continue
  if (contaminant_lcs_.groupby(['sim']).apply(lambda x: x['passband'].nunique()) < defs.n_phot).any(): continue

  k += 1
  epsilon = defs.epsilon_ / k**(1./n) # to be GLIE
  for use in range(defs.reuse): #each episode reused, epsilon decayed after
    t = time.time()
    R_tau = 0
    KN_lc = KN_lc_.copy()
    contaminant_lcs = contaminant_lcs_.copy()
    for timestep in range(1, defs.horizon): #FIXME: in SARSA action prime does not exist, can we just return reward then?
      state = pythia.State(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep)
      e_greedy_action = pythia.e_greedy(state, w, epsilon)
      reward = pythia.get_reward(e_greedy_action, KN_idx)
      state_prime, KN_lc, contaminant_lcs, obs = pythia.next_state(KN_lc, KN_idx,
                                                        contaminants, contaminant_lcs, contaminant_idx,
                                                        timestep+1, e_greedy_action)
      e_greedy_action_prime = pythia.e_greedy(state_prime, w, epsilon)

      if timestep != defs.horizon:
        td0 = reward + gamma * pythia.lin_VFA(state_prime, e_greedy_action_prime, w)[0,0]
      else:
        td0 = reward
      delw = (td0 - pythia.lin_VFA(state, e_greedy_action, w)[0,0]) * pythia.feat_state_action(state, e_greedy_action)
      w[:-1, 0], w[-1, 0] = adam.update(k + 1, w=w[:-1, 0], b=w[-1, 0], dw=delw[:-1, 0], db=delw[-1, 0]) #FIXME: t not updated until after reused
      R_tau += reward
    print(k, R_tau, np.linalg.norm(w, ord=1))
    train_status = pd.concat([train_status,
                            pd.DataFrame([[k, np.linalg.norm(w, ord=1), R_tau, time.time()-t]],
                            columns=['k', '||w||', 'R_tau', 'comp_time'])
                          ])
    train_status.to_csv(dest+'/train_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.csv', index=False)
    with open(dest+'/train_linw_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.npy', 'wb') as f:
      np.save(f, w)
    with open(dest+'/adam_obj_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.pkl', 'wb') as f:
      pickle.dump(adam, f, pickle.HIGHEST_PROTOCOL)
  if epsilon < 0.05 and train_status.groupby('k').mean()['R_tau'].rolling(window=50).mean().iloc[-1] > 2.5*(defs.horizon-1)/defs.N:
    with open(dest+'/train_linw_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'_'+str(k)+'.npy', 'wb') as f:
      np.save(f, w)
