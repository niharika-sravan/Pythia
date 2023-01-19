import numpy as np
import pandas as pd
import random
import math
import warnings

from scipy import stats
from scipy import interpolate
from keras.applications.xception import preprocess_input

import utils
import defs

class State():
  def __init__(self, KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep):
    self.KN_lc = KN_lc[KN_lc['mjd'] - KN_lc['tc'] < timestep]
    self.KN_idx = KN_idx
    self.contaminants = contaminants
    self.contaminant_lcs = contaminant_lcs[contaminant_lcs['mjd'] - contaminant_lcs['tc'] < timestep]
    self.contaminant_idx = contaminant_idx
    self.timestep = timestep

def e_greedy(state, w, epsilon):
  Q = np.zeros((np.shape(defs.action_set)[1], 1))
  for i in range(np.shape(defs.action_set)[1]): #for all actions
    Q[i, 0] = lin_VFA(state, defs.action_set[:, i][:, np.newaxis], w)
  if random.random() < epsilon:
    e_greedy_action = defs.action_set[:, random.randrange(np.shape(defs.action_set)[1])]
  else:
    e_greedy_action = defs.action_set[:, random.choice(np.argwhere(Q == np.amax(Q))[:, 0])]
  return e_greedy_action[:, np.newaxis]

def lin_VFA(state, action, w):
  #Q = np.dot(np.vstack((np.tile(state, np.shape(action)[1]), action)).T, w)
  Q = np.dot(feat_state_action(state, action).T, w)
  return Q

def feat_state_action(state, action):
  event_select = get_event_id(action)
  filter_select = get_filter_id(action)
  KN_lc = state.KN_lc.copy()
  contaminant_lcs = state.contaminant_lcs.copy()
  KN_lc = KN_lc[KN_lc['mag_err'] != np.inf] #deals with mag_err that is []
  contaminant_lcs = contaminant_lcs[contaminant_lcs['mag_err'] != np.inf] #deals with mag_err that is []
  KN_lc['mag_err'] = KN_lc['mag_err'].astype(float)
  contaminant_lcs['mag_err'] = contaminant_lcs['mag_err'].astype(float)

  if event_select == state.KN_idx:
    if not KN_lc.empty:
      try:
        tc = KN_lc['tc'].sample().item()
        sim = KN_lc['sim'].sample().item()
        phot_pred = pd.DataFrame([[tc+state.timestep+1, filter_select, sim, tc]], columns = ['mjd', 'passband', 'sim', 'tc'])
        phot_pred = fit_GP(KN_lc, phot_pred)
        KN_lc = pd.concat([KN_lc, phot_pred])
      except ValueError:
        pass
  else:
    cont_lc = contaminant_lcs[contaminant_lcs['sim'] == state.contaminants[state.contaminant_idx.index(event_select)]]
    if not cont_lc.empty:
      try:
        tc = cont_lc['tc'].sample().item()
        sim = cont_lc['sim'].sample().item()
        phot_pred = pd.DataFrame([[tc+state.timestep+1, filter_select, sim, tc]], columns = ['mjd', 'passband', 'sim', 'tc'])
        phot_pred = fit_GP(cont_lc, phot_pred)
        contaminant_lcs = pd.concat([contaminant_lcs, phot_pred])
      except ValueError:
        pass

  repr = np.zeros((defs.Xception_h, defs.Xception_w, defs.n_filt))
  repr = get_repr(KN_lc, state.KN_idx, repr)
  for i in range(defs.N - 1):
    cont_lc = contaminant_lcs[contaminant_lcs['sim'] == state.contaminants[i]]
    repr = get_repr(cont_lc, state.contaminant_idx[i], repr)

  feat = defs.model.predict(preprocess_input(np.expand_dims(repr, axis=0))).flatten()
  return np.vstack((feat[:, np.newaxis], 1))

def fit_GP(lc, GP_lc):
  GP_lc = utils.GP_free((lc['mjd']-lc['tc']).values,
                         lc['mag'].values,
                         lc['mag_err'].values,
                         lc['passband'].values,
                         GP_lc,
                         1.)
  return GP_lc

def get_repr(lc, idx, rpr):
  def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.)/(2 * np.power(sig, 2.)))

  for i in range(defs.n_filt):
    lc_filt = lc[lc['passband'] == i+1]
    if lc_filt.empty: continue
    bin_means, bin_edges, binnumber = stats.binned_statistic(lc_filt['mjd'] - lc_filt['tc'],
                                                            lc_filt['mag'],
                                                            statistic = 'mean',
                                                            bins = defs.s,
                                                            range = [0, defs.horizon])
    bin_mag = np.nan_to_num(bin_means)
    bin_means, bin_edges, binnumber = stats.binned_statistic(lc_filt['mjd'] - lc_filt['tc'],
                                                            lc_filt['mag_err'],
                                                            #sigma of addition of two sigmas
                                                            statistic = lambda x: np.sqrt(np.sum(np.power(x, 2)))/len(x),
                                                            bins = defs.s,
                                                            range = [0, defs.horizon])
    bin_mag_err = np.nan_to_num(bin_means)
    start_x = int(defs.s * (idx%np.sqrt(defs.N)) + defs.pad)
    start_y = int(defs.s * math.floor(idx/np.sqrt(defs.N)) + defs.pad)
    for j in range(defs.s):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rpr[start_y:start_y+defs.s, start_x+j, i] = gaussian(defs.mag_bins, bin_mag[j], bin_mag_err[j])
  return rpr

def get_event_id(action):
  return math.floor(int(np.nonzero(action[:, 0])[0])/defs.n_filt)

def get_filter_id(action):
  return int(np.nonzero(action[:, 0])[0])%defs.n_filt + 1

def get_reward(action, KN_idx):
  '''
  rate = fade_rate(state)
  if fade < 0.3:
    reward = +1
  elif fade > 0.3:
    reward = -1
  else: #implicity is a nan so false for above
    reward = 0
  '''
  # if photo was for KN +1 else 0
  if get_event_id(action) == KN_idx:
    reward = 1
  else:
    reward = 0
  return reward

def get_photo(lc_all, lc, time_obs, filter):
  sim = lc['sim'].sample().item()
  tc = lc['tc'].sample().item()
  time = time_obs + tc
  obs = pd.DataFrame([[time, filter]], columns = ['mjd', 'passband'])
  ids = sim.split('_')
  if ids[0] not in list(utils.types.keys()): id_ = ids[0] + '_' + ids[1] + '_' + ids[3]
  else: id_ = sim
  lc_filt = lc_all[(lc_all['sim'] == id_) & (lc_all['passband'] == filter)]
  f = interpolate.interp1d(lc_filt['mjd'], lc_filt['mag'], fill_value="extrapolate") # extrap needed for some Ia sims
  obs['mag'] = f(obs['mjd'])
  obs['mag_err'] = np.nan
  def set_lim(obs, lim):
    if (obs['mag'] > lim).item():
      obs['mag'] = lim
      obs['mag_err'] = np.inf
  if filter == 1:
      set_lim(obs, float(defs.lim1_ToO.sample()))
  elif filter == 2:
      set_lim(obs, float(defs.lim2_ToO.sample()))
  else: #FIXME: using same as r
      set_lim(obs, float(defs.lim2_ToO.sample()))
  if (obs['mag_err'] != np.inf).item():
    obs = utils.estimate_mag_err(obs)
  obs['sim'] = sim
  obs['tc'] = tc
  obs['survey'] = 'Pythia'
  return obs

def next_state(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep, action):
  # add action random during timestep (could be same bin as survey)
  event_select = get_event_id(action)
  filter_select = get_filter_id(action)
  time_obs = random.uniform(timestep-1, timestep)
  if event_select == KN_idx:
    obs = get_photo(defs.KN_lc_all, KN_lc, time_obs, filter_select)
    KN_lc = pd.concat([KN_lc, obs]) # contain action photometry
  else:
    cont_lc = contaminant_lcs[contaminant_lcs['sim'] == contaminants[contaminant_idx.index(event_select)]]
    obs = get_photo(defs.contaminant_lc_all, cont_lc, time_obs, filter_select)
    contaminant_lcs = pd.concat([contaminant_lcs, obs]) # contain action photometry
  state_prime = State(KN_lc, KN_idx, contaminants, contaminant_lcs, contaminant_idx, timestep)
  return state_prime, KN_lc, contaminant_lcs, obs
