# importing this has not been tested since the function was moved out of utils.
# some vars may not point correctly

import sncosmo
import sfdmap
dustmap = sfdmap.SFDMap('../aug/sfddata-master')

import utils

def fit_Ia_model(meta,LC,spec_z=False):
  flux,flux_err=utils.get_flux(LC)
  data=Table()
  if spec_z: data.meta['z']=float(meta['z'])
  data['mjd']=LC['mjd'].tolist()
  data['band']=LC['passband'].map(utils.sncosmo_band_name_dict).tolist()
  data['flux']=flux.tolist()
  data['flux_err']=flux_err.tolist()
  data['zp']=LC['passband'].map(utils.ZTF_zp_dict).tolist()
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
