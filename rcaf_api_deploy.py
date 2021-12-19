import pandas as pd                                                       
import numpy as np                                                                                                            
from scipy import signal
import json
from biosppy.signals import ecg   
from biosppy.signals import eeg
from biosppy.signals import resp
from scipy.interpolate import interp1d   
import pickle
import joblib                             
from sklearn.metrics import log_loss,f1_score   
from timeit import default_timer as timer
import flask
from flask import Flask, jsonify, request

import lightgbm as lgb

app = Flask(__name__)



################################################################################



def interpolation_fn(timestamps,biosppy_ts, biosppy_values):
    """linear interpolation function to produce heart rate, resp rate all time steps"""
    interpolation = interp1d(biosppy_ts,biosppy_values, kind="linear", fill_value="extrapolate")  
    return interpolation(timestamps)

def noise_free(data,w):

  ''' function takes raw  signal and removes some noise present init gives noise free signal''' 
  n=5
  b,a = signal.butter(n,w,fs=256)

  return signal.filtfilt(b,a,data)

def biosppy(df):
    """THIS FUNCTION WILL DERIVE ALL FEATURE THAT IS GENEARTED USING BIOSPPY MODULE"""
                      
    df['filt_ecg'] = noise_free(df.ecg,100)                    # filtering ecg signal 
    df['filt_respiration'] = noise_free(df.r,0.7)              # filtering r signal 
  
    bio=ecg.ecg(df["ecg"],sampling_rate=256,show=False)                                           #heart rate from ecg
    df["heart_rate"]=interpolation_fn(df["time"],bio["heart_rate_ts"],bio["heart_rate"])
    
    
    bio=resp.resp(df["r"],sampling_rate=256,show=False)                                             #resp rate from r signal
    df["resp_rate"]=interpolation_fn(df["time"],bio["resp_rate_ts"],bio["resp_rate"])
    
       
    return df

def potential_differences(df):
  """FUNCTION TO CALCULATE POTENTIAL DIFFERENCE BETWEEN ELECTRODES"""
      
  df['fp1_f7'] = df['eeg_fp1'] - df['eeg_f7']
  df['f7_t3'] = df['eeg_f7'] - df['eeg_t3']
  df['t3_t5'] = df['eeg_t3'] - df['eeg_t5']
  df['t5_o1'] = df['eeg_t5'] - df['eeg_o1']
  df['fp1_f3'] = df['eeg_fp1'] - df['eeg_f7']
  df['f3_c3'] = df['eeg_f3'] - df['eeg_c3']
  df['c3_p3'] = df['eeg_c3'] - df['eeg_p3']
  df['p3_o1'] = df['eeg_p3'] - df['eeg_o1']

  df['fz_cz'] = df['eeg_fz'] - df['eeg_cz']
  df['cz_pz'] = df['eeg_cz'] - df['eeg_pz']                     # train potential differences 
  df['pz_poz'] = df['eeg_pz'] - df['eeg_poz']

  df['fp2_f8'] = df['eeg_fp2'] - df['eeg_f8']
  df['f8_t4'] = df['eeg_f8'] - df['eeg_t4']
  df['t4_t6'] = df['eeg_t4'] - df['eeg_t6']
  df['t6_o2'] = df['eeg_t6'] - df['eeg_o2']
  df['fp2_f4'] = df['eeg_fp2'] - df['eeg_f4']
  df['f4_c4'] = df['eeg_f4'] - df['eeg_c4']
  df['c4_p4'] = df['eeg_c4'] - df['eeg_p4']
  df['p4_o2'] = df['eeg_p4'] - df['eeg_o2']

  
  return df

features_n = ['fp1_f7', 'f7_t3', 't3_t5', 't5_o1', 'fp1_f3', 'f3_c3', 'c3_p3', 'p3_o1', 'fz_cz', 'cz_pz',
                'pz_poz', 'fp2_f8', 'f8_t4', 't4_t6', 't6_o2', 'fp2_f4', 'f4_c4', 'c4_p4', 'p4_o2', 'resp_rate','heart_rate', "gsr",'filt_ecg','filt_respiration']

##################################################################################

@app.route('/')
def hello_world():
  return 'Hello World!'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction_func1():
  ''' taking 1 datapoint as input with 27 features and returning the predicted output for it '''

  start = timer()
  train=pickle.load(open('train_df.pkl','rb')) # sample 5000
  train=train.drop('event',axis=1)

  to_predict_list = request.form.to_dict()
  raw_data = to_predict_list['raw_data']
  raw_data = list(raw_data.split(','))
  if raw_data[1] == "'LOFT'":
    raw_data[1]=4
  elif raw_data[1] == "'CA'":
    raw_data[1]=0
  elif raw_data[1] == "'DA'":
    raw_data[1]=1
  elif raw_data[1] == "'SS'":
    raw_data[1]=3

  for i in range(len(raw_data)):

    if i==0 or i==3:
      raw_data[i] = int(raw_data[i])
    elif i==1 :
      raw_data[i] = str(raw_data[i])
    else:
      raw_data[i] = float(raw_data[i]) 

  
  raw_data=np.array(raw_data,dtype=float)
  raw_data=raw_data.reshape(1,27)
  raw_data=pd.DataFrame(raw_data,columns=train.columns.tolist())
  raw_data=raw_data.append(train)
  raw_data = raw_data.reset_index() 

  raw_data=biosppy(raw_data)
  raw_data=potential_differences(raw_data)
  model= pickle.load(open('lightgbm.pkl','rb')) 
  prob = model.predict_proba(raw_data[features_n]) 

  end = timer()
  print('total time : ',end - start)

  cls = ['baseline','SS','CA','DA']
  prob = prob[0].tolist()

  return jsonify({'predicted_probability :':list(zip(cls,prob)),'total time :':end - start})  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

