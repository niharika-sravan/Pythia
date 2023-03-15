import numpy as np
import pandas as pd
import random

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import defs

dest = 'outdir/n9'

alpha_list = [1e-2, 1e-3]
gamma_list = [0.9, 0.5, 0.1]
n_list = [3.]
smooth = 50
rows,cols = len(alpha_list), len(gamma_list)

for n in n_list:
  fig = plt.figure(figsize=(8,4))#(4, 2.5))
  fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.98, wspace=0.15, hspace=0.6)
  gs = gridspec.GridSpec(rows, cols)
  sax = []
  for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))

  n_episodes = int((0.99/0.05)**n)

  R_tau = np.zeros(n_episodes)
  for i in range(np.shape(R_tau)[0]):
    R_rand = 0
    for j in range(1, defs.horizon): #SARSA agent is not estimating reward in timestep 6 because no action at 7
      if random.randrange(0, defs.N) == 0: R_rand += 1
    R_tau[i] = R_rand

  for i,alpha in enumerate(alpha_list):
    for j,gamma in enumerate(gamma_list):
        train_status = pd.read_csv(dest+'/train_'+str(alpha)+'_'+str(gamma)+'_'+str(n)+'.csv')
        idx = len(gamma_list)*i + j
        avg_score = train_status.groupby('k').mean()['R_tau'].rolling(window=smooth).mean()
        sigma_score = train_status.groupby('k').mean()['R_tau'].rolling(window=smooth).std()
        sax[idx].plot(range(n_episodes), pd.Series(R_tau).rolling(window=smooth).mean(),
                      lw = 0.5,
                      color = '#ff7f0e',
                      label = 'Random Agent')
        sax[idx].fill_between(range(n_episodes),
                              pd.Series(R_tau).rolling(window=smooth).mean()+pd.Series(R_tau).rolling(window=smooth).std(),
                              pd.Series(R_tau).rolling(window=smooth).mean()-pd.Series(R_tau).rolling(window=smooth).std(),
                              alpha = 0.1, color = '#ff7f0e', edgecolor = None)
        sax[idx].plot(train_status.groupby('k').groups.keys(),
                      avg_score,
                      label = 'Pythia')
        sax[idx].fill_between(train_status.groupby('k').groups.keys(),
                      avg_score+sigma_score, avg_score-sigma_score,
                      alpha = 0.2)
        print(alpha, gamma, n, avg_score.max())
        #sax[idx].set_title(r'$\gamma=$'+str(gamma)+r'; $\alpha=$'+str(alpha), fontsize = 10)
        sax[idx].set_xlim(0, 8500)
        sax[idx].set_ylim(0, 6.1)
        sax[idx].axhline(y = 6, c='#2ca02c', ls = 'dashdot')
        sax[idx].axhline(y = (defs.horizon-1)/defs.N, c='#ff7f0e', ls = 'dashdot')
        sax[idx].axvline(x = (1/0.05)**n, c = 'grey', ls = '--', lw = 0.7)
        sax[idx].axvline(x = (1/0.1)**n, c = 'grey', ls = '--', lw = 0.7)
        sax[idx].axvline(x = (1/0.2)**n, c = 'grey', ls = '--', lw = 0.7)
        sax[idx].axvline(x = (1/1)**n, c = 'grey', ls = '--', lw = 0.7)
        sax[idx].set_xlabel('episodes')
        sax[idx].set_ylabel('score')
        sax[idx].text(155, 2.2, '20% explore', rotation = 90, fontsize = 6)
        sax[idx].text(1030, 2.2, '10% explore', rotation = 90, fontsize = 6)
        sax[idx].text(8030, 2.2, '5% explore', rotation = 90, fontsize = 6)
        sax[idx].text(450, 0.3, 'Avg Random Score', fontsize=6, color = '#ff7f0e')
        sax[idx].text(500, 5.7, 'Max Score', fontsize=6, color = '#2ca02c')

  sax[idx].legend(fontsize = 7, loc = 1)

  fig.savefig(dest+'/hyper_'+str(n)+'.png', dpi=300)

# print(pd.read_csv('data/n2_thin_bright/train_0.01_0.5_2.0.csv').tail(n=(defs.horizon-1)*smooth).to_string())
