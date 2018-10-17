# you can test the Score codes here, and copy this code out and save as Score.py file

import json
import numpy as np
import pandas as pd
import os
import pickle

feature_cols = ['cycle_norm', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
feature_cols = [s for s in feature_cols if s not in ['setting3', 's1', 's5', 's10', 's16', 's18', 's19']]


def getTestInput(): # return [?, 26] matrix
    dataColumns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    test_df = pd.read_csv('upload/test_FD001.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = dataColumns
    return test_df

def normalizeInputData(test_df): # input [?, 26], output [?, 20]
    test_df['cycle_norm'] = test_df['cycle']
    cols_normalize = test_df.columns.difference(['id','cycle'])
    with open('min_max_scaler.pickle','rb') as f:
        min_max_scaler = pickle.load(f)
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)
    return test_df[['id','cycle']+feature_cols]


def testRun(engine_id=3): # engine 20 is failing, but engine 3 is not at the time of test
    test_df = getTestInput()  # [?, 26]
    test_df = test_df[test_df['id']==engine_id] # ? filtered by id
    #normalize
    test_df = normalizeInputData(test_df)  # [?, 20]
    # feature engineering
    lag_window = 5
    lag_cols = [s for s in feature_cols if s not in ['cycle_norm','setting1','setting2','setting3']]
    # build lagging features - train data set
    df_mean = test_df[lag_cols].rolling(window=lag_window).mean()
    df_std = test_df[lag_cols].rolling(window=lag_window).std()
    df_mean.columns = ['MA'+s for s in lag_cols]
    df_std.columns = ['STD'+s for s in lag_cols]
    df_input = pd.concat([test_df,df_mean,df_std], axis=1, join='inner')
    input_array = df_input.values[-1:,2:]

    with open('model_Linear.pickle','rb') as f:
        model = pickle.load(f)
    pred_test = model.predict(input_array)
    print('prediction: ', pred_test[0])
    
    
def init():
    global model
    #ws = Workspace.from_config()
    #model = Model(ws, "model_Linear")
    #model.download(target_dir = '.')
    with open('model_Linear.pickle','rb') as f:
        model = pickle.load(f)

def run(test_json):
    test_df = pd.read_json(test_json, orient='split') # [?, 26] filtered by id

    #normalize
    test_df = normalizeInputData(test_df)  # [?, 20]
    # feature engineering
    lag_window = 5
    lag_cols = [s for s in feature_cols if s not in ['cycle_norm','setting1','setting2','setting3']]
    # build lagging features - train data set
    df_mean = test_df[lag_cols].rolling(window=lag_window).mean()
    df_std = test_df[lag_cols].rolling(window=lag_window).std()
    df_mean.columns = ['MA'+s for s in lag_cols]
    df_std.columns = ['STD'+s for s in lag_cols]
    df_input = pd.concat([test_df,df_mean,df_std], axis=1, join='inner')
    input_array = df_input.values[-1:,2:]

    pred_test = model.predict(input_array)
    return json.dumps(int(pred_test[0]))
       
def testRunJSON(engine_id=3):  
    test_df = getTestInput()  # [?, 26]
    test_df = test_df[test_df['id']==engine_id] # ? filtered by id
    test_df = test_df.tail(5)
    
    test_json = test_df.to_json(orient='split')
    # ... networking, pretend call web service ...
    return run(test_json)