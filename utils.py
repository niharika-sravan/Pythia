import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import sncosmo
#import sfdmap
#dustmap = sfdmap.SFDMap('../aug/sfddata-master')
import george

window=7
ZTF_zp_dict={1:26.325,2:26.275,3:25.660} #from ZSDS_explanatory pg 67
central_wvl={1:471.70,2:653.74,3:687.20} #for ZTF
band_colors={1:'limegreen',2:'orangered',3:'goldenrod'}#,'limegreen','darkturquoise',
band_name_dict={0:'u',1:'g',2:'r',3:'i',4:'z',5:'y'}
sncosmo_band_name_dict={1:'ztfg',2:'ztfr',3:'ztfi'}

from astropy.cosmology import FlatLambdaCDM
cosmo=FlatLambdaCDM(H0=70,Om0=0.3,Tcmb0=2.725)

types={
      'Ia':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'Ia'},
      'II':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'GP'},
      'IIn':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'GP'},
      'IIb':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'GP'},
      'Ib':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'GP'},
      'Ic':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'GP'},
      'Ic-BL':{'logz_bins':np.arange(-2.8,-0.79,0.2),'method':'GP'},
      'SLSN-I':{'logz_bins':np.arange(-1.8,-0.39,0.2),'method':'GP'},
      'SLSN-II':{'logz_bins':np.arange(-1.8,-0.39,0.2),'method':'GP'},
}

uncer_params=pd.read_pickle('data/uncer_params_forced.pkl')

def get_band_info(survey) -> list:
  """
  returns band names for each survey
  """
  if survey=='ZTF_public':
    band_list=[1,2]
  elif survey=='ZTF_full':
    band_list=[1,2,3]
  return band_list

def get_flux(LC):
  """
  helper function to get fluxes from LC magnitudes using ZTF zero-points
  """
  flux=10.**((LC['mag']-LC['passband'].map(ZTF_zp_dict))/-2.5)
  flux_err=abs(flux*LC['mag_err']*(np.log(10.)/2.5))
  return flux,flux_err

def fit_Ia_model(meta,LC,spec_z=False):
  flux,flux_err=get_flux(LC)
  data=Table()
  if spec_z: data.meta['z']=float(meta['z'])
  data['mjd']=LC['mjd'].tolist()
  data['band']=LC['passband'].map(sncosmo_band_name_dict).tolist()
  data['flux']=flux.tolist()
  data['flux_err']=flux_err.tolist()
  data['zp']=LC['passband'].map(ZTF_zp_dict).tolist()
  data['zpsys']=['ab']*len(flux)
  c=SkyCoord(meta['R.A.'],meta['Declination'],
            unit=(u.hourangle, u.deg))
  dust=sncosmo.CCM89Dust() #r_v=A(V)/E(B−V); A(V) is total extinction in V
  model=sncosmo.Model(source='salt2',
                      effects=[dust],
                      effect_names=['mw'],
                      effect_frames=['obs'])
  ebv=dustmap.ebv(c) #ICRS frame
  if not spec_z:
    model.set(mwebv=ebv)
    result,fitted_model=sncosmo.fit_lc(data,model,
                                      ['z','t0','x0','x1','c'],
                                      bounds={'z':(0.,0.2)})
  else:
    model.set(z=data.meta['z'],mwebv=ebv)  # set the model's redshift.
    result,fitted_model=sncosmo.fit_lc(data,model,
                                      ['t0','x0','x1','c'])

  return result,fitted_model

def GP_free(time,mag,mag_err,passband,LC_GP,k_corr_scale):
  def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(mag)

  def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(mag)

  dim=[central_wvl[int(pb)] for pb in passband]
  signal_to_noise_arr=(np.abs(mag)/
                      np.sqrt(mag_err**2+(1e-2*np.max(mag))**2))
  scale=np.abs(23-mag[signal_to_noise_arr.argmax()])
  # setup GP model
  kernel=((0.5*scale)**2*george.kernels.ExpSquaredKernel([10**2,300**2], ndim=2))
  kernel.freeze_parameter('k1:log_constant') #Fixed overall scale
  x_data=np.vstack([time,dim]).T
  gp=george.GP(kernel,mean=23)
  # train
  gp.compute(x_data,mag_err)
  result=minimize(neg_ln_like,
                  gp.get_parameter_vector(),
                  jac=grad_neg_ln_like)
  gp.set_parameter_vector(result.x)
  #predict
  ts_pred=np.zeros((LC_GP.shape[0],2))
  ts_pred[:,0]=LC_GP['mjd']
  ts_pred[:,1]=LC_GP['passband'].map(central_wvl)*k_corr_scale
  LC_GP['mag'],mag_var=gp.predict(mag,ts_pred,return_var=True)
  LC_GP['mag_err']=np.sqrt(mag_var)
  return LC_GP

def estimate_mag_err(df):
  df['mag_err']=df.apply(
                lambda x: (uncer_params['band']==x['passband']) &
                          (pd.arrays.IntervalArray(
                            uncer_params['interval']).contains(x['mag'])),
                          axis=1).apply(
                lambda x: stats.skewnorm.rvs(
                            uncer_params[x]['a'],
                            uncer_params[x]['loc'],
                            uncer_params[x]['scale']
                                            ), axis=1)
  return df

def estimate_GP_mag(df,LC,band_list,sim_err=True,kernel='SE'):
  GP=pd.DataFrame(data=np.vstack((np.repeat(df['mjd'].values,len(band_list)),
                                      np.tile(band_list,len(df['mjd'].values))
                                     )).T,columns=['mjd','passband'])
  GP=GP.drop_duplicates()
  if kernel=='SE':
    GP=GP_free(LC['mjd'].values,
                   LC['mag'].values,
                   LC['mag_err'].values,
                   LC['passband'].values,
                   GP,
                   1.)
  if kernel=='SM':
    GP=GP_SM(LC['mjd'].values,
                   LC['mag'].values,
                   LC['mag_err'].values,
                   LC['passband'].values,
                   GP,
                   1.)
  if sim_err: GP=estimate_mag_err(GP)
  df=pd.merge(df,GP,how='left',
              left_on=['mjd','passband'],right_on=['mjd','passband'])
  return df

class AdamOptim():
  def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.m_dw, self.v_dw = 0, 0
    self.m_db, self.v_db = 0, 0
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.alpha = alpha
  def update(self, t, w, b, dw, db):
    ## dw, db are from current minibatch
    ## momentum beta 1
    # *** weights *** #
    self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
    # *** biases *** #
    self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

    ## rms beta 2
    # *** weights *** #
    self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
    # *** biases *** #
    self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

    ## bias correction
    m_dw_corr = self.m_dw/(1-self.beta1**t)
    m_db_corr = self.m_db/(1-self.beta1**t)
    v_dw_corr = self.v_dw/(1-self.beta2**t)
    v_db_corr = self.v_db/(1-self.beta2**t)

    ## update weights and biases
    w = w + self.alpha*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
    b = b + self.alpha*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
    return w, b
