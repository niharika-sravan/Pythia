import pandas as pd
import numpy as np
import random

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

band_colors={'g':'limegreen','r':'orangered','i':'goldenrod'}#,'limegreen','darkturquoise',

for type in ['Ia','II','IIn','IIb','Ib','Ic','Ic-BL','SLSN-I','SLSN-II']:
  lcs=pd.read_csv('data/'+type+'.csv')
  lcs=lcs.replace(np.inf,np.nan).dropna()
  print(type, lcs.groupby('sim').ngroups)

  fig = plt.figure(figsize=(6, 6))
  fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.98, wspace=0.3, hspace=0.3)
  rows, cols=3, 2
  gs = gridspec.GridSpec(rows,cols)
  sax = []
  for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))

  hist, bins = np.histogram(lcs.groupby(['sim']).apply(lambda x: x.shape[0]),bins=np.arange(-0.5,10.,1))
  for name,group in lcs.groupby(['sim','passband']).apply(lambda x: x.shape[0]).groupby(level=1):
    hist, bins = np.histogram(group,bins=np.arange(-0.5,10.,1))
    sax[0].step(bins, np.pad(hist, (1, 0)) / hist.max(), alpha=0.5, color=band_colors[name],lw=2)
  hist, bins = np.histogram(lcs.groupby(['sim']).apply(lambda x: x.shape[0]),bins=np.arange(-0.5,10.,1))
  sax[0].step(bins, np.pad(hist, (1, 0)) / hist.max(),color='k',lw=0.5)

  for name,group in lcs.groupby(['sim','passband']).apply(lambda x: x['mag'].min()).groupby(level=1):
    hist, bins = np.histogram(group,bins=np.arange(16,22,0.5))
    sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(), alpha=0.5, color=band_colors[name],lw=2)
  hist, bins = np.histogram(lcs.groupby(['sim']).apply(lambda x: x['mag'].min()),bins=np.arange(16,22,0.5))
  sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(),color='k',lw=0.5)

  for name,group in lcs.groupby(['sim','passband']).apply(lambda x: (x['mjd']-x['tc'])[x['mag'].idxmin()]).groupby(level=1):
    hist, bins = np.histogram(group,bins=np.arange(0,4.5,0.2))
    sax[2].step(bins, np.pad(hist, (1, 0)) / hist.max(), alpha=0.5, color=band_colors[name],lw=2)
  hist, bins = np.histogram(lcs.groupby(['sim']).apply(lambda x: (x['mjd']-x['tc'])[x['mag'].idxmin()]),bins=np.arange(0,4.5,0.2))
  sax[2].step(bins, np.pad(hist, (1, 0)) / hist.max(),color='k',lw=0.5)

  #lcs.groupby(['sim','passband']).apply(lambda x: (x['mjd']-x['tc'])[x['mag'].idxmin()]).groupby(level=1).apply(lambda x: sax[2].hist(x,alpha=0.3,color=band_colors[x.name]))

  sax[0].set_xlabel('num photo')
  sax[0].get_yaxis().set_visible(False)
  sax[1].set_xlabel('peak mag')
  sax[1].get_yaxis().set_visible(False)
  sax[2].set_xlabel('time max since KN trigger')
  sax[2].get_yaxis().set_visible(False)

  sax[3].scatter(lcs.groupby(['sim']).apply(lambda x: x.shape[0]),lcs.groupby(['sim']).apply(lambda x: x['mag'].min()),s=1)
  sax[3].set_xlabel('num photo (all)')
  sax[3].set_ylabel('peak mag (all)')
  sax[3].invert_yaxis()

  for name,group in lcs.groupby(['sim','passband']).apply(lambda x: (x['mjd']-x['tc'])[x['mjd'].idxmin()]).groupby(level=1):
    hist, bins = np.histogram(group,bins=np.arange(0,4.5,0.2))
    sax[4].step(bins, np.pad(hist, (1, 0)) / hist.max(), alpha=0.5, color=band_colors[name],lw=2)
  hist, bins = np.histogram(lcs.groupby(['sim']).apply(lambda x: x['mjd'].min()-x['tc'][x['mjd'].idxmin()]),bins=np.arange(0,4.5,0.2))
  sax[4].step(bins, np.pad(hist, (1, 0)) / hist.max(),color='k',lw=0.5)

  sax[4].set_xlabel('time first detection since KN trigger')
  sax[4].get_yaxis().set_visible(False)

  fig.savefig('data/'+type+'_stats.png',dpi=300)

  fig = plt.figure(figsize=(10,10))
  fig.subplots_adjust(left=0.07,right=0.98,bottom=0.07,top=0.98,wspace=0.,hspace=0.)
  rows,cols=6,5
  gs = gridspec.GridSpec(rows,cols)
  sax = []
  for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))

  sim_list=random.sample(lcs['sim'].unique().tolist(),rows*cols)
  s_bool={True:4,False:2}
  i=0
  for sim,LC in lcs[lcs['sim'].isin(sim_list)].groupby('sim'):
    for name, group in LC.groupby(['passband','ToO']):
      sax[i].errorbar(group['mjd']-group['tc'],group['mag'],yerr=group['mag_err'].astype(float),
                      fmt='o',ms=s_bool[name[1]],c=band_colors[name[0]])
    sax[i].set_xlabel('time since KN trigger')
    sax[i].set_xlim(0.,8.)
    sax[i].set_ylim(22.,12.)
    if i%cols==0: sax[i].set_ylabel('mag')
    else: sax[i].set_yticklabels([])
    i+=1

  fig.savefig('data/'+type+'_sample.png',dpi=300)
  plt.close()

  #### for all ####
  lcs=pd.read_csv('data/'+type+'_all.csv')

  fig = plt.figure(figsize=(6,6))
  fig.subplots_adjust(left=0.02,right=0.98,bottom=0.1,top=0.98,wspace=0.3,hspace=0.3)
  rows,cols=3,2
  gs = gridspec.GridSpec(rows,cols)
  sax = []
  for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))

  for name,group in lcs.groupby(['sim','passband']).apply(lambda x: x['mag'].min()).groupby(level=1):
    if not name in band_colors.keys(): continue
    hist, bins = np.histogram(group,bins=np.arange(16,22,0.5))
    sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(), alpha=0.5, color=band_colors[name],lw=2)
  hist, bins = np.histogram(lcs[lcs['passband'].isin(band_colors.keys())].groupby(['sim']).apply(lambda x: x['mag'].min()),bins=np.arange(16,22,0.5))
  sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(),color='k',lw=0.5)

  for name,group in lcs.groupby(['sim','passband']).apply(lambda x: (x['mjd']-x['mjd'].min())[x['mag'].idxmin()]).groupby(level=1):
    if not name in band_colors.keys(): continue
    hist, bins = np.histogram(group,bins=np.arange(0,4.5,0.5))
    sax[2].step(bins, np.pad(hist, (1, 0)) / hist.max(), alpha=0.5, color=band_colors[name],lw=2)
  hist, bins = np.histogram(lcs[lcs['passband'].isin(band_colors.keys())].groupby(['sim']).apply(lambda x: (x['mjd']-x['mjd'].min())[x['mag'].idxmin()]),bins=np.arange(0,4.5,0.5))
  sax[2].step(bins, np.pad(hist, (1, 0)) / hist.max(),color='k',lw=0.5)

  sax[1].set_xlabel('peak mag')
  sax[1].get_yaxis().set_visible(False)
  sax[2].set_xlabel('time max mag - time min')
  sax[2].get_yaxis().set_visible(False)

  heatmap, xedges, yedges = np.histogram2d(lcs[lcs['passband'].isin(band_colors.keys())].groupby(['sim']).apply(lambda x: x['mjd'][x['mag']<22.].max()-x['mjd'][x['mag'].idxmin()]).fillna(0),
                            lcs[lcs['passband'].isin(band_colors.keys())].groupby(['sim']).apply(lambda x: x['mag'].min()),
                            bins=50)
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
  sax[5].imshow(heatmap.T, extent=extent, origin='lower')
  #sax[5].scatter(lcs[lcs['filter'].isin(band_colors.keys())].groupby(['sim']).apply(lambda x: x['jd'][x['mag']<22.].max()-x['jd'][x['mag'].idxmin()]),
  #      lcs[lcs['filter'].isin(band_colors.keys())].groupby(['sim']).apply(lambda x: x['mag'].min()),s=1)
  sax[5].set_xlabel('time max - time max mag')
  sax[5].set_ylim(22.,16.)
  sax[5].set_ylabel('peak mag')

  fig.savefig('data/'+type+'_stats_all.png',dpi=300)

  if type in ['BNS', 'NSBH', 'GRB']:
    for i,sim in enumerate(sim_list):
      sim_list[i]=sim.split('_')[0]+'_'+sim.split('_')[2]+'_all'

  fig = plt.figure(figsize=(10,10))
  fig.subplots_adjust(left=0.07,right=0.98,bottom=0.07,top=0.98,wspace=0.,hspace=0.)
  rows,cols=6,5
  gs = gridspec.GridSpec(rows,cols)
  sax = []
  for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols*r+c]))

  i=0
  for sim,LC in lcs[lcs['sim'].isin(sim_list)].groupby('sim'):
    #if sim != sorted(sim_list)[i]: i+=1
    for name,group in LC.groupby(['passband']):
      if name in band_colors.keys(): sax[i].plot(group['mjd']-group['mjd'].min(),group['mag'],
                      c=band_colors[name[0]])
      else: sax[i].scatter(group['mjd']-group['mjd'].min(),group['mag'],s=2,label=name)
    sax[i].set_xlabel('time since KN trigger')
    sax[i].set_xlim(0.,8.)
    sax[i].set_ylim(22.,12.)
    if i%cols==0: sax[i].set_ylabel('mag')
    else: sax[i].set_yticklabels([])
    i+=1
  sax[0].legend()

  fig.savefig('data/'+type+'_sample_all.png',dpi=300)
  plt.close()
