import numpy as np
import pandas as pd
import random

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import defs

dest = 'outdir/n9'

alpha_list = [1e-0, 1e-1, 1e-2]
gamma_list = [0.9, 0.5, 0.1]
n_list = [2.]
smooth = 50

for n in n_list:
  fig = plt.figure(figsize=(12, 6))
  fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.96, wspace=0.15, hspace=0.6)
  rows,cols = len(alpha_list), len(gamma_list)
  gs = gridspec.GridSpec(rows, cols)
  sax = []
  for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))

  n_episodes = int((0.99/0.01)**n)

  R_tau = np.zeros(n_episodes)
  for i in range(np.shape(R_tau)[0]):
    R_rand = 0
    for j in range(1, defs.horizon-1): #SARSA agent is not estimating reward in timestep 6 because no action at 7
      if random.randrange(0, defs.N) == 0: R_rand += 1
    R_tau[i] = R_rand

  for i,alpha in enumerate(alpha_list):
    for j,gamma in enumerate(gamma_list):
        train_status = pd.read_csv(dest+'/train_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.csv')
        idx = len(gamma_list)*i + j
        avg_score = train_status.groupby('k').mean()['R_tau'].rolling(window=smooth).mean()
        sigma_score = train_status.groupby('k').mean()['R_tau'].rolling(window=smooth).std()
        sax[idx].plot(range(n_episodes), pd.Series(R_tau).rolling(window=smooth).mean(), lw = 0.2, color='#ff7f0e')
        sax[idx].fill_between(range(n_episodes),
                              pd.Series(R_tau).rolling(window=smooth).mean()+pd.Series(R_tau).rolling(window=smooth).std(),
                              pd.Series(R_tau).rolling(window=smooth).mean()-pd.Series(R_tau).rolling(window=smooth).std(),
                              alpha=0.1, color='#ff7f0e', edgecolor = None)

        sax[idx].plot(train_status.groupby('k').groups.keys(),
                      avg_score,
                      label = 'Pythia')
        sax[idx].fill_between(train_status.groupby('k').groups.keys(),
                      avg_score+sigma_score, avg_score-sigma_score,
                      alpha = 0.3)
        print(alpha, gamma, n, avg_score.max())
        sax[idx].set_title(r'$\gamma=$'+str(gamma)+r'; $\alpha=$'+str(alpha), fontsize=10)
        sax[idx].set_xlim(0, 1000)
        sax[idx].set_ylim(0, 3.5)
        sax[idx].axhline(y = (defs.horizon-2)/defs.N, c='#ff7f0e', label = 'Random Agent')
        sax[idx].axvline(x = (1/0.05)**n, c='grey', ls = '--', lw=0.7)
        sax[idx].axvline(x = (1/0.1)**n, c='grey', ls = '--', lw=0.7)
        sax[idx].axvline(x = (1/1)**n, c='grey', ls = '--', lw=0.7)
        sax[idx].set_xlabel('episodes')
        sax[idx].set_ylabel('score')
  sax[0].text(5, 1.1, '100% explore', rotation = 90, fontsize=6)
  sax[0].text(105, 1.2, '10% explore', rotation = 90, fontsize=6)
  sax[0].text(405, 1.2, '5% explore', rotation = 90, fontsize=6)
  sax[0].legend(fontsize=8)

  fig.savefig(dest+'/hyper_'+str(n)+'.png', dpi=300)

# print(pd.read_csv('data/n2_thin_bright/train_0.01_0.5_2.0.csv').tail(n=(defs.horizon-1)*smooth).to_string())
