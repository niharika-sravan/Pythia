# will overwrite all files
# TODO: save SALT2 model params

import json
import pandas as pd
import numpy as np
import math
import os,sys,glob,time
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from scipy import stats
from joblib import dump, load
from astropy.coordinates import SkyCoord
import sfdmap
dustmap = sfdmap.SFDMap('../aug/sfddata-master')

import utils
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("typ", type=str)
parser.add_argument("nevents", type=int)
args=parser.parse_args()

revisit=load('data/revisit_kde_public.joblib')
revisit_i=load('data/revisit_kde_i.joblib')
sampling=pd.read_pickle('data/sampling_public.pkl')
lim1=load('data/lims_public_1.joblib')
lim2=load('data/lims_public_2.joblib')
lim3=load('data/lims_i.joblib')
noise=load('data/noise_kde_gri.joblib')

def resimulate_Ia(meta,LC,sim,z_new,ra_new,dec_new):
  success=True
  try:
    result,fitted_model=utils.fit_Ia_model(meta,LC,spec_z=True)
    if result.chisq>300.: success=False
  except RuntimeError:
    success=False
  if success:
    #put model in new conditions
    c=SkyCoord(ra_new,dec_new,unit='deg')
    ebv=dustmap.ebv(c) #ICRS frame
    fitted_model.set(z=z_new,mwebv=ebv)
    zp=sim['passband'].map(utils.ZTF_zp_dict)
    flux=fitted_model.bandflux(sim['passband'].map(utils.sncosmo_band_name_dict),
                              sim['mjd'],
                              zp=zp,
                              zpsys='ab')
    sim['mag']=zp-2.5*np.log10(flux+flux*noise.sample(n_samples=sim.shape[0])[:,0])
  return sim,success

def resimulate_GP(meta,LC,sim,z_new,*args):
  success=True
  z_org=float(meta['z'])
  # sample GP fit at contracted timestamps and redshifted mean wvls
  obs_src_time=LC['mjd'].min()+(LC['mjd']-LC['mjd'].min())/(1.+z_org)
  sim_src_time=sim['mjd'].min()+(sim['mjd']-sim['mjd'].min())/(1.+z_new)
  sim=utils.GP_free(obs_src_time.values,
                    LC['mag'].values,
                    LC['mag_err'].values,
                    LC['passband'].values,
                    sim,
                    k_corr_scale=(1.+z_org)/(1.+z_new))
  #dim/brighten source
  sim['mag']=sim['mag']+5.*np.log10(utils.cosmo.luminosity_distance(z_new)/
                              utils.cosmo.luminosity_distance(z_org))
  return sim,success

sn_list=[]
for sn in glob.glob('data/gen_set/*_meta.json'):
  LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
  if LC[LC['mjd']<LC['mjd'][LC['mag'].idxmin()]].shape[0]==0: continue #conservative elimination of events with no rise
  with open(sn,'r') as f: meta=json.load(f)
  if meta['Type']==args.typ: sn_list.append(sn)
train=pd.DataFrame()
uniform=pd.DataFrame()
inj_stats=pd.DataFrame()
sim_stats=pd.DataFrame()
nbins=len(utils.types[args.typ]['logz_bins'])-1
events_per_bin=math.ceil(args.nevents/nbins)
simnum=0
for exp in ['180','300']:
  sampling_ToO=pd.read_pickle('data/sampling_ToO_'+exp+'.pkl')
  lim1_ToO=load('data/lims_Too_'+exp+'_1.joblib')
  lim2_ToO=load('data/lims_Too_'+exp+'_2.joblib')
  for i in range(nbins):
    ctr=0
    while ctr<events_per_bin:
      #pick random event
      sn=np.random.choice(sn_list)
      with open(sn,'r') as f: meta=json.load(f)
      LC=pd.read_json(sn.split('_meta')[0]+'.json',orient='index')
      #simulate new environment
      z_new=10.**np.random.uniform(utils.types[args.typ]['logz_bins'][i],utils.types[args.typ]['logz_bins'][i+1])
            #+np.random.normal(0, 0.001))
      ra_new=360*np.random.random_sample()
      dec_new=np.random.uniform(-30, 90)
      inj_stats=pd.concat([inj_stats,
                          (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                          z_new,ra_new,dec_new,
                                          sn.split('_meta')[0].split('/')[-1]]],
                                        columns=['z_org','ra_org','dec_org',
                                                'z_new','ra_new','dec_new','src_event']))
                          ])
      #get new observing strategy
      sim=pd.DataFrame()
      start=np.random.uniform(LC['mjd'].min()-7.,
                              LC['mjd'][LC['mag'].idxmin()]+7.) # after ~peak
      t=np.random.uniform(start,start+2)
      while t-start<utils.window:
        sample=sampling.sample()
        sim=pd.concat([sim,pd.DataFrame(np.array([t+sample['t'].values[0],
                                        sample['bands'].values[0]]).T)
                      ])
        t+=float(revisit.sample())
      t=np.random.uniform(start,start+4)
      while t-start<utils.window:
        sim=pd.concat([sim,pd.DataFrame([[t,3]])])
        t+=float(revisit_i.sample())
      sim['ToO']=False
      sim_ToO=pd.DataFrame()
      t=np.random.uniform(start,start+1)
      too_samps=sampling_ToO.sample(np.random.choice([1,2]))
      for ind,too in too_samps.iterrows():
          sim_ToO=pd.concat([sim_ToO,pd.DataFrame(np.array([t+too['t'],
                                                    too['bands']]).T)
                            ])
          t+=1
      sim_ToO['ToO']=True
      sim=pd.concat([sim,sim_ToO])
      sim=(sim.rename(columns={0:'mjd',1:'passband'})
              .sort_values(by=['mjd'])
              .reset_index(drop=True))
      #simulate mag
      sim,success=globals()['resimulate_'+utils.types[args.typ]['method']
                            ](
                          meta,LC,sim,z_new,ra_new,dec_new)
      if not success: continue
      sim['mag_err']=np.nan #resets mag_err set by GP when using it and adds it for Ia
      def set_lim(sim,row,lim):
        if row['mag']>lim:
          sim.loc[row.name,'mag']=lim
          sim.loc[row.name,'mag_err']=np.inf
      for filt,group in sim.groupby('passband'):
        for idx,row in group.iterrows():
          if filt==1 and not row['ToO']:
              set_lim(sim,row,float(lim1.sample()))
          elif filt==1 and row['ToO']:
              set_lim(sim,row,float(lim1_ToO.sample()))
          elif filt==2 and not row['ToO']:
              set_lim(sim,row,float(lim2.sample()))
          elif filt==2 and row['ToO']:
              set_lim(sim,row,float(lim2_ToO.sample()))
          else:
              set_lim(sim,row,float(lim3.sample()))
      sim=utils.estimate_mag_err(sim[sim['mag_err']!=np.inf])
      if not sim['mag_err'].astype(bool).all():
        #to omit cases where mag_err is []
        continue

      sim_LC=sim.replace(np.inf,np.nan).dropna()
      if sim_LC.empty: continue #otherwise cannot do idxmin later
      flux=10.**((sim_LC['mag']-sim_LC['passband'].map(utils.ZTF_zp_dict))/-2.5)
      sim_LC['SNR']=flux/abs(flux*sim_LC['mag_err']*(np.log(10.)/2.5))
      first_phot=sim_LC.loc[sim_LC['mjd'].idxmin()]
      if (first_phot['SNR']<3.): continue

      #sample uniformly
      band_list=utils.get_band_info('ZTF_full')
      times=np.arange(0,7,0.1)+start
      unf=pd.DataFrame(data=np.vstack((np.repeat(times,len(band_list)),
                                        np.tile(band_list,len(times))
                                       )).T,columns=['mjd','passband'])
      unf,success=globals()['resimulate_'+utils.types[args.typ]['method']
                            ](
                          meta,LC,unf,z_new,ra_new,dec_new)
      if not success: continue

      sim_stats=pd.concat([sim_stats,
                            (pd.DataFrame([[meta['z'],meta['R.A.'],meta['Declination'],
                                          z_new,ra_new,dec_new,
                                          sim['mag'].min(),
                                          sn.split('_meta')[0].split('/')[-1]]],
                                          index=[simnum],
                                          columns=['z_org','ra_org','dec_org',
                                                  'z_new','ra_new','dec_new',
                                                  'peak_mag','src_event']))
                          ])
      sim['sim']=str(simnum)+'_'+exp
      sim['tc']=start
      sim['luminosity_distance']=utils.cosmo.luminosity_distance(z_new)
      train=pd.concat([train,sim])
      unf['sim']=str(simnum)+'_'+exp
      unf['tc']=start
      unf['luminosity_distance']=utils.cosmo.luminosity_distance(z_new)
      uniform=pd.concat([uniform,unf])
      ctr+=1
      simnum+=1
train['passband']=train['passband'].map({1:'g',2:'r',3:'i'})
train.reset_index(drop=True).to_csv('data/'+args.typ+'.csv',index=False)
uniform['passband']=uniform['passband'].map({1:'g',2:'r',3:'i'})
uniform.reset_index(drop=True).to_csv('data/'+args.typ+'_all.csv',index=False)
inj_stats.to_pickle('data/'+args.typ+'_inj_stats.pkl')
sim_stats.to_pickle('data/'+args.typ+'_sim_stats.pkl')

#stats
fig=plt.figure()
plt.hist(inj_stats['z_org'],alpha=0.5,bins=10.**utils.types[args.typ]['logz_bins'],label='inj')
plt.hist(sim_stats['z_new'],alpha=0.5,bins=10.**utils.types[args.typ]['logz_bins'],label='sim')
plt.xlabel('log z')
plt.xscale('log')
plt.ylabel('count')
plt.legend()
fig.savefig('data/'+args.typ+'_z_dist.png')
