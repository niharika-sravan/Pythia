import numpy as np
import pandas as pd
import os, sys, glob, shutil
import json
import matplotlib.pyplot as plt
from joblib import dump, load
import requests
import time

from sklearn.neighbors import KernelDensity
from scipy import stats
from scipy.stats import chi2_contingency
from pandas.plotting import scatter_matrix

# STEP 1: get all BTS LCs
#all ZTF SNe, from BTS website, peak before Aug 31
ZTF_SNe=pd.read_csv('../ZTF/ZTF_SNe.csv')

# get all photometry for BTS SNe from Kowalski
from penquins import Kowalski

username = "nsravan@astro.caltech.edu"
password = "hgvw0601bu88v3vdgxnpjsmnjvognvfm"
protocol = "https"

k = Kowalski(username=username,
     password=password,
     protocol=protocol)

def get_data(object_id):
  q = {"query_type": "find",
      "query": {"catalog": "ZTF_alerts",
                "filter": {"objectId": object_id},
                "projection": {"_id": 0,
                              "candid": 1,
                              "candidate.jd": 1,
                              "candidate.fid": 1,
                              "candidate.magpsf": 1,
                              "candidate.sigmapsf": 1}
                }
      }

  r = k.query(query=q)
  df=pd.DataFrame()
  for phot in r['data']:
    df=pd.concat([df,pd.DataFrame([[phot['candidate']['jd']-2400000.5,
                                    phot['candidate']['fid'],
                                    phot['candidate']['magpsf'],
                                    phot['candidate']['sigmapsf']]],
                                    columns=['mjd','passband','mag','mag_err'])
                  ])
  if not df.empty:
    with open('data/ZTF_SNe/'+object_id+'.json','w') as f:
      json.dump(df.reset_index(drop=True).to_dict(orient='index'),f,indent=4)
  return

for idx, sn in ZTF_SNe.iterrows():
  get_data(sn['ZTFID'])
  meta={'z':sn['redshift'],'Type':sn['type'],
        'R.A.':sn['RA'],'Declination':sn['Dec']}
  with open('data/ZTF_SNe/'+sn['ZTFID']+'_meta.json', 'w') as f:
    json.dump(meta,f,indent=4)

'''
DO NOT USE FOR BULK
import requests

token='c0de54d5-8963-4bf5-943e-d84b35bc1765'
response = requests.get(
          "https://fritz.science/api/alerts/ZTF18abdfazk",
          headers={'Authorization': f'token {token}'},
)

response.text
'''

# STEP 2: QC for BTS and historic
# change all IIP, IIL to II
for fldr in ['data/ZTF_SNe']:
  for sn in glob.glob(fldr+'/*_meta.json'):
    with open(sn,'r') as f:
      meta=json.load(f)
    try:
      meta['Type']=meta['Type'].split('SN ')[1]
      if (meta['Type']=='IIP' or meta['Type']=='IIL'):
        meta['Type']='II'
      with open(sn,'w') as f:
        json.dump(meta,f,indent=4)
    except:
      pass

# dropping less than 4 data points or data in only one band
classes=['Ia','II','IIb','Ib','IIn','Ic','Ic-BL','SLSN-I','SLSN-II']
LC_char=pd.DataFrame()
for sn in glob.glob('data/ZTF_SNe/*_meta.json'):
  with open(sn,'r') as f: meta=json.load(f)
  LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  LC=LC.sort_values(by=['mjd'],kind='mergesort')
  if (LC.shape[0]<4 or LC['passband'].nunique()<3 or meta['Type'] not in classes):
    continue
  LC_char=LC_char.append({
            'max_gap_early':LC[LC['mjd']<LC['mjd'].min()+90]['mjd'].diff().max(),
            'phot_den':LC.shape[0]/(LC['mjd'].max()-LC['mjd'].min()),
            'n_rise':LC[LC['mjd']<LC['mjd'][LC['mag'].idxmin()]].shape[0], #conservative
            'z':float(meta['z']),
            'type':meta['Type'],
            'ra':meta['R.A.'],
            'dec':meta['Declination'],
            'name':sn.split('_meta')[0].split('/')[1]},
            ignore_index=True)

#dropping no rise or early gaps >30
LC_char=LC_char[(LC_char['n_rise']>0) & (LC_char['max_gap_early']<30)]
LC_char=LC_char.set_index('name')
scatter_matrix(LC_char)
print(LC_char.describe())
print(LC_char.groupby(['type']).size())

for name, group in LC_char.groupby('type'):
  ax=np.log10(group['z']).hist(alpha=0.2,log=True,bins=np.arange(-2.7,-0.4,0.2),label=name)
  print(np.log10(group['z']).min(), np.log10(group['z']).max(), name)
ax.legend()

for sn in LC_char.index:
  shutil.copyfile('data/ZTF_SNe/'+sn+'.json','data/gen_set/'+sn+'.json')
  shutil.copyfile('data/ZTF_SNe/'+sn+'_meta.json','data/gen_set/'+sn+'_meta.json')

# STEP 3: noise model to apply to SN Ia

'''
#explore thresh
fits=pd.DataFrame()
for sn in sn_list:
  try:
    result,fitted_model=fit_model(sn)
    fits=pd.concat([fits,pd.DataFrame([[sn,result.chisq]])])
  except (RuntimeError, ValueError) as e:
    continue

for i, row in fits[(fits[1]>100) & (fits[1]<500)].sort_values(by=1).iterrows():
  result,fitted_model=fit_model(row[0])
  LC=sncosmo.read_lc('sncosmo/'+row[0]+'.json',format='json')
  sncosmo.plot_lc(LC,model=fitted_model,errors=result.errors,
                figtext=[sn+'\nchi_sq='+str(result.chisq)]);
'''

import utils

band_name_dict={1:'ztfg',2:'ztfr',3:'ztfi'}
ZTF_zp_dict={1:26.325,2:26.275,3:25.660} #from ZSDS_explanatory pg 67
chi_sq_thresh=300

err_df=pd.DataFrame()
for sn,props in LC_char.iterrows():
  if (sn.startswith('ZTF') and props['type']=='Ia'):
    LC=pd.read_json('data/gen_set/'+sn+'.json',orient='index')
    LC=LC[LC['mjd']<LC['mjd'].min()+60.]
    with open('data/gen_set/'+sn+'_meta.json','r') as f:
      meta=json.load(f)  #at least one gri band in first 90 days
    if not LC['passband'].nunique()==len(band_name_dict.keys()): continue
    try: result,fitted_model=utils.fit_Ia_model(meta,LC,spec_z=True)
    except RuntimeError: continue
    if result.chisq<chi_sq_thresh:
      LC['flux_sncosmo']=fitted_model.bandflux(LC['passband'].map(band_name_dict),
                                        LC['mjd'],
                                        zp=LC['passband'].map(ZTF_zp_dict),
                                        zpsys='ab')
      flux=10.**((LC['mag']-LC['passband'].map(ZTF_zp_dict))/-2.5)
      err_df=pd.concat([err_df,
                      pd.DataFrame([LC['mag'],
                                    (flux-LC['flux_sncosmo'])/LC['flux_sncosmo']]).T])

err_df=err_df.rename(columns={'Unnamed 0':'noise'})
errs=err_df[~err_df['noise'].isin([np.nan, np.inf, -np.inf])]['noise'].values

bw=1e-2
bins=np.arange(-0.5,0.5,0.001)
fig, ax = plt.subplots()
plt.hist(errs,bins=bins,density=True)
kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(errs[:,None])
x_plot=bins[:,np.newaxis]
dens=np.exp(kde.score_samples(x_plot))
plt.plot(x_plot[:,0],dens,lw=0.5)
plt.savefig('data/noise_gaussian_gri_'+str(bw)+'.pdf',dpi=300)

dump(kde,'data/noise_kde_gri.joblib')

###
# This portion deals with observing style
###

'''
import pyvo.dal
import time

client=pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')

for jd in np.arange(58849+2400000.5,59549+2400000.5):
  print(jd)
  obstable=client.search("""
  SELECT field,rcid,fid,expid,obsjd,exptime,maglimit,ipac_gid
  FROM ztf.ztf_current_meta_sci WHERE (obsjd BETWEEN {0} AND {1})
  """.format(jd,jd+1)).to_table()
  names=('obsjd','field','rcid','fid','ipac_gid')
  renames=('jd','fieldid','chid','filterid','programid')
  obstable.rename_columns(names,renames)
  with open('../ZTF/allobs/'+str(jd)+'.csv','w') as f:
    obstable.to_pandas().to_csv(f,index=False)
  time.sleep(1)

'''

obstable=pd.DataFrame()
for jd in np.arange(59184+2400000.5,59549+2400000.5):
  with open('../ZTF/allobs/'+str(jd)+'.csv','r') as f:
    obstable=pd.concat([obstable,pd.read_csv(f)])

obstable['jd']=obstable['jd']-2400000.5
obstable=obstable.rename(columns={'jd':'mjd'}).reset_index(drop=True)
public=obstable[obstable['programid']==1]

pp_i=obstable[obstable['filterid']==3]
pp_i=pp_i[pp_i['exptime']==30]

# STEP 4: estimate revisit intervals for ZTF classified
# does not account for gaps due to moon

df=pd.DataFrame()
for field, group in public.groupby('fieldid'):
  group=group[group['chid']==np.random.choice(group['chid'].unique())]
  rev=group[(group['mjd'].diff()>1.) | (group['mjd'].diff().isna())
              ][['mjd','filterid']].rename(columns={'mjd':'gap'})
  rev['gap']=rev['gap'].diff()
  rev=rev.fillna(0)
  rev=rev[rev['gap']!=0.]
  df=pd.concat([df,rev])

bw=1e-2
visit=df['gap'].values
bins=np.arange(0.,10.,0.01)
plt.figure()
plt.hist(visit,bins=bins,density=True)
kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(visit[:,None]) #set visually
x_plot=bins[:,np.newaxis]
dens=np.exp(kde.score_samples(x_plot))
plt.plot(x_plot[:,0],dens,lw=0.5)
plt.xlabel('days')
plt.savefig('data/revisit_'+str(bw)+'.pdf',dpi=300)

dump(kde,'data/revisit_kde_public.joblib')

df=pd.DataFrame()
for field, group in pp_i.groupby('fieldid'):
  group=group[group['chid']==np.random.choice(group['chid'].unique())]
  rev=group[(group['mjd'].diff()>1.) | (group['mjd'].diff().isna())
              ][['mjd','filterid']].rename(columns={'mjd':'gap'})
  rev['gap']=rev['gap'].diff()
  rev=rev.fillna(0)
  rev=rev[rev['gap']!=0.]
  df=pd.concat([df,rev])

bw=1e-2
visit=df['gap'].values
bins=np.arange(0.,10.,0.01)
plt.figure()
plt.hist(visit,bins=bins,density=True)
kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(visit[:,None]) #set visually
x_plot=bins[:,np.newaxis]
dens=np.exp(kde.score_samples(x_plot))
plt.plot(x_plot[:,0],dens,lw=0.5)
plt.xlabel('days')
plt.savefig('data/revisit_i_'+str(bw)+'.pdf',dpi=300)

dump(kde,'data/revisit_kde_i.joblib')

# STEP 5: verify independence of per visit attributes and record observing pattern per visit
# build contingency table (revisit finished 0-4 hrs Fig 9 Bellm 2019)
'''
df_cont=pd.DataFrame()
for sn in glob.glob('train_lcs/ZTF*_meta.json'):
  with open(sn,'r') as f:
    meta=json.load(f)
  df_LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  df_LC['mjd']=round(df_LC['mjd']-7/24) #putting mid observation at mn to help with rounding
  df_strat=pd.concat([group['passband'].reset_index(drop=True)
                        for name, group in df_LC.groupby('mjd')],
                        axis=1).T
  df_strat=df_strat.replace({1.0:'r',2.0:'g'})#,np.nan:'none'})
  df_strat=pd.get_dummies(df_strat)
  df_cont=pd.concat([df_cont,
                  df_strat.groupby(df_strat.reset_index(drop=True).index%2).sum()])

df_cont=df_cont.groupby(df_cont.reset_index(drop=True).index%2).sum()
chi2_contingency(df_cont)
# I cannot accept they are dependent
'''

df_samp=pd.DataFrame()
for field, group in public.groupby('fieldid'):
  group=group[group['chid']==np.random.choice(group['chid'].unique())]
  group['mjd_round']=round(group['mjd']-7/24) #putting mid observation at mn to help with rounding
  df_samp=pd.concat([df_samp,
                    pd.DataFrame([[(g['mjd']-g['mjd'].min()).values,
                                  g['filterid'].values]
                        for n, g in group.groupby('mjd_round')])
                    ])
df_samp.rename(columns=
              {0:'t',1:'bands'}
              ).reset_index(drop=True).to_pickle('data/sampling_public.pkl')

#sampling for ToO

#GW200105 - UT 2020-01-07 + UT 2020-01-08 - 180s
#GW200115 - 2020-01-15 - 300s

# GRB afterglows
# GRB200514B 10:01 UT on 2020 May 14, 59075 - 300s
# GRB200826A 09:22:55 UT on 2020 Aug 26, 59087 - 300s
# GRB210510A 05:02:53.10 UT on 2021-05-11, 59345 - 180s
# GRB210529B 05:06:36.18 UT on 2021-05-30, 59364 + 05:07:50.96 UT on 2021-05-31, 59365 - 180s

obstable=pd.DataFrame()
for jd in np.arange(58849+2400000.5,59549+2400000.5):
  with open('../ZTF/allobs/'+str(jd)+'.csv','r') as f:
    obstable=pd.concat([obstable,pd.read_csv(f)])
obstable['jd']=obstable['jd']-2400000.5
obstable=obstable.rename(columns={'jd':'mjd'}).reset_index(drop=True)

large=obstable[(obstable['programid']==2) &
                (obstable['exptime']==180) &
                (round(obstable['mjd'].astype(float)).isin([58855,58856,59345,59364,59365]))]

small=obstable[(obstable['programid']==2) &
                (obstable['exptime']==300) &
                (round(obstable['mjd'].astype(float)).isin([58863,58983,59087]))]

df_samp=pd.DataFrame()
for field, group in small.groupby('fieldid'):
  group=group[group['chid']==np.random.choice(group['chid'].unique())]
  group['mjd_round']=round(group['mjd']-7/24) #putting mid observation at mn to help with rounding
  df_samp=pd.concat([df_samp,
                    pd.DataFrame([[(g['mjd']-g['mjd'].min()).values,
                                  g['filterid'].values]
                        for n, g in group.groupby('mjd_round')])
                    ])
df_samp.rename(columns=
              {0:'t',1:'bands'}
              ).reset_index(drop=True).to_pickle('data/sampling_ToO_300.pkl')

df_samp=pd.DataFrame()
for field, group in large.groupby('fieldid'):
  group=group[group['chid']==np.random.choice(group['chid'].unique())]
  group['mjd_round']=round(group['mjd']-7/24) #putting mid observation at mn to help with rounding
  df_samp=pd.concat([df_samp,
                    pd.DataFrame([[(g['mjd']-g['mjd'].min()).values,
                                  g['filterid'].values]
                        for n, g in group.groupby('mjd_round')])
                    ])
df_samp.rename(columns=
              {0:'t',1:'bands'}
              ).reset_index(drop=True).to_pickle('data/sampling_ToO_180.pkl')

# STEP 6: build photometry uncertainity model (function of seeing, sky brightness, instrument+)
# binned by mag; see mag_bins
#csv comes from schoty.py

df=pd.read_csv('../ZTF/all_forced.csv')
df_uncer=df[df['mag']!=99]
df_uncer=df_uncer.replace({'g':1,'r':2,'i':3})

mag_bins=[12.,18.,20.,21.,23.]
bins=pd.cut(df_uncer['mag'], bins=mag_bins)
param_df=pd.DataFrame()
for name, group in df_uncer.groupby([bins,'filter']):
  uncer=group['mag_unc'].values
  p=group['mag_unc'].quantile(0.98)
  pbins=np.arange(0.,0.5,0.0025)
  ae,loce,scalee=stats.skewnorm.fit(group[(group['mag_unc']<p)]['mag_unc'])
  p=stats.skewnorm.pdf(pbins,ae,loce,scalee)
  plt.figure()
  plt.hist(uncer,bins=pbins,density=True)
  plt.plot(pbins,p,lw=2.)
  plt.xlabel('magnitude uncertainity')
  plt.savefig('data/uncer_skew_'+str(name)+'.pdf',dpi=300)
  param_df=pd.concat([param_df,pd.DataFrame([[name[1],name[0],ae,loce,scalee]])]
                    )

param_df.rename(columns={0:'band',1:'interval',2:'a',3:'loc',4:'scale'}
                ).reset_index(drop=True).to_pickle('data/uncer_params_forced.pkl')

# STEP 7: fit mag lim in each band for public (30), pp_i (30), large (180), small (300)
#this file is from Igor
'''
Sure, I repeated the query making sure that the MJDs of interest are included. I also included FIELD and PROGRAMID information.
Thank you! Would it be possible to only put data in the range 58849-59549 in addition to GRB ToO mjds?
I expanded the JD range to include the extra data you need.

> I tried matching the records Igor sent to that from TAP and found more entries in TAP.
>
> Here's an example:
> ZTF pointing history from TAP with pid 1 in date range 59184 to 59549 has 5.7M records.
> The file Igor sent with pid 1 in date range 59184 to 59549 has 35k records
>
> Since there a unique record per ZTF pointing from TAP for each chip, I tried to select only record per pointing with the hypothesis that there is only one mag limit per field in forced photometry (as opposed to in TAP that has different mag limits for each chip).
>
> The down sampling (1 record per pointing; random chip) reduces ZTF public pointing history to 92.5k records.
>
> So it looks like there are about 57.5k more records on TAP for this date range for the public survey even with my conservative estimate. I couldn't come up with where else the discrepancy could be coming from so have gone ahead and worked with the matches that I found.
'''
forced_lim=pd.read_csv('../ZTF/forced_limmag.csv')
band_name_dict={'g':1,'r':2,'i':3}

forced_lim['jd']=forced_lim['jd']-2400000.5
forced_lim['filter']=forced_lim['filter'].map(band_name_dict)
forced_lim=forced_lim.rename(columns={'filter':'filterid','jd':'mjd'}).reset_index(drop=True)
public_forced=forced_lim[forced_lim['programid']==1]
pp_i_forced=forced_lim[(forced_lim['programid']==2) & (forced_lim['filterid']==3)]
pp_forced=forced_lim[forced_lim['programid']==2]

#merge public gr 30s in yr range with public_forced (has some extra random dates)
public_forced_meta=public.groupby('mjd').sample().merge(public_forced[['limmag','mjd']],how='left',on='mjd')
public_forced=public_forced_meta[~public_forced_meta['limmag'].isna()]

for band, group in public_forced.groupby('filterid'):
  bw=1e-1
  # on each mjd for each field get mag lim for one chip?
  # all chips have diff mag lim
  #group=group.sample(int(group.shape[0]/100))
  lims=group['limmag'].values
  bins=np.arange(12.,24.,0.01)
  plt.figure()
  plt.hist(lims,bins=bins,density=True)
  kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(lims[:,None]) #set visually
  x_plot=bins[:,np.newaxis]
  dens=np.exp(kde.score_samples(x_plot))
  plt.plot(x_plot[:,0],dens,lw=0.5)
  plt.xlabel('mag limit')
  plt.savefig('data/lims_public_'+str(band)+'_'+str(bw)+'.pdf',dpi=300)

  dump(kde,'data/lims_public_'+str(band)+'.joblib')

ppi_forced_meta=pp_i.groupby('mjd').sample().merge(pp_i_forced[['limmag','mjd']],how='left',on='mjd')
pp_i_forced=ppi_forced_meta[~ppi_forced_meta['limmag'].isna()]

bw=1e-1
lims=pp_i_forced['limmag'].values
bins=np.arange(14.,24.,0.01)
plt.figure()
plt.hist(lims,bins=bins,density=True)
kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(lims[:,None]) #set visually
x_plot=bins[:,np.newaxis]
dens=np.exp(kde.score_samples(x_plot))
plt.plot(x_plot[:,0],dens,lw=0.5)
plt.xlabel('mag limit')
plt.savefig('data/lims_i.pdf',dpi=300)

dump(kde,'data/lims_i.joblib')

small_forced_meta=small.groupby('mjd').sample().merge(pp_forced[['limmag','mjd']],how='left',on='mjd')
small_forced=small_forced_meta[~small_forced_meta['limmag'].isna()]

for band, group in small_forced.groupby('filterid'):
  bw=1e-1
  lims=group['limmag'].values
  bins=np.arange(19.,23.,0.05)
  plt.figure()
  plt.hist(lims,bins=bins,density=True)
  kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(lims[:,None]) #set visually
  x_plot=bins[:,np.newaxis]
  dens=np.exp(kde.score_samples(x_plot))
  plt.plot(x_plot[:,0],dens,lw=0.5)
  plt.xlabel('mag limit')
  plt.savefig('data/lims_small_'+str(band)+'_'+str(bw)+'.pdf',dpi=300)

  dump(kde,'data/lims_Too_300_'+str(band)+'.joblib')

large_forced_meta=large.groupby('mjd').sample().merge(pp_forced[['limmag','mjd']],how='left',on='mjd')
large_forced=large_forced_meta[~large_forced_meta['limmag'].isna()]

for band, group in large_forced.groupby('filterid'):
  bw=1e-1
  lims=group['limmag'].values
  bins=np.arange(20.,23.,0.05)
  plt.figure()
  plt.hist(lims,bins=bins,density=True)
  kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(lims[:,None]) #set visually
  x_plot=bins[:,np.newaxis]
  dens=np.exp(kde.score_samples(x_plot))
  plt.plot(x_plot[:,0],dens,lw=0.5)
  plt.xlabel('mag limit')
  plt.savefig('data/lims_large_'+str(band)+'_'+str(bw)+'.pdf',dpi=300)

  dump(kde,'data/lims_Too_180_'+str(band)+'.joblib')
