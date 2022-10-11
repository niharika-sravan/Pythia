import numpy as np
import pandas as pd
import math
import joblib

class train_data():
  def __init__(self, status):
    self.KN_lc = pd.read_csv('data/KNe_'+status+'.csv')
    self.contaminant_lc = pd.read_csv('data/contaminant_'+status+'.csv')
    #remove SBO
    self.contaminant_lc = self.contaminant_lc[~self.contaminant_lc['sim'].str.startswith('SBO')]

    self.KN_lc['passband'] = self.KN_lc['passband'].map({'g':1, 'r':2, 'i':3})
    self.contaminant_lc['passband'] = self.contaminant_lc['passband'].map({'g':1, 'r':2, 'i':3})

    self.KN_lc['ToO'] = self.KN_lc['ToO'].map({True:'ZTF_ToO', False:'ZTF'})
    self.contaminant_lc['ToO'] = self.contaminant_lc['ToO'].map({True:'ZTF_ToO', False:'ZTF'})
    self.KN_lc.rename(columns={'ToO': 'survey'}, inplace = True)
    self.contaminant_lc.rename(columns={'ToO': 'survey'}, inplace = True)

KN_lc_all = pd.read_csv('data/KNe_all.csv')
contaminant_lc_all = pd.read_csv('data/contaminant_all.csv')
KN_lc_all['passband'] = KN_lc_all['passband'].map({'g':1, 'r':2, 'i':3})
contaminant_lc_all['passband'] = contaminant_lc_all['passband'].map({'g':1, 'r':2, 'i':3})

N = 9 # make sure sq root is integer
horizon = 7
n_filt = 3
action_set = np.identity(n_filt * N)
n_phot = 1
reuse = horizon-2 # hyperparamter to tune: how many times to go over same episode

from keras.applications.xception import Xception
model = Xception(weights='imagenet', include_top=False, pooling='max')
Xception_h, Xception_w = 299, 299
out_size = 2048

pad = math.floor(Xception_w%np.sqrt(N)/2)
s = int((Xception_w-Xception_w%np.sqrt(N))/ np.sqrt(N))
mag_bins = np.linspace(13, 23, s) #size of viewing window, hard set

lim1_ToO = joblib.load('data/lims_Too_300_1.joblib')
lim2_ToO = joblib.load('data/lims_Too_300_2.joblib')

epsilon_ = 1
