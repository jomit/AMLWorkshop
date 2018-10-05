import pandas as pd
import numpy as np
import keras

from PM_Prep import train_df, test_df, feature_cols, printScores

'''
Traditional feature engieering
'''
lag_window = 5
lag_cols = [s for s in feature_cols if s not in ['cycle_norm','setting1','setting2','setting3']]

# build lagging features - train data set
df_mean = train_df[lag_cols].rolling(window=lag_window).mean()
df_std = train_df[lag_cols].rolling(window=lag_window).std()
df_mean.columns = ['MA'+s for s in lag_cols]
df_std.columns = ['STD'+s for s in lag_cols]
df_train = pd.concat([train_df,df_mean,df_std], axis=1, join='inner')
''' df_train columns: 
        'id', 'cycle', 'RUL', 'label1', 'cycle_norm', 'setting1', 'setting2',
       's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14',
       's15', 's17', 's20', 's21', 'MAs2', 'MAs3', 'MAs4', 'MAs6', 'MAs7',
       'MAs8', 'MAs9', 'MAs11', 'MAs12', 'MAs13', 'MAs14', 'MAs15', 'MAs17',
       'MAs20', 'MAs21', 'STDs2', 'STDs3', 'STDs4', 'STDs6', 'STDs7', 'STDs8',
       'STDs9', 'STDs11', 'STDs12', 'STDs13', 'STDs14', 'STDs15', 'STDs17',
       'STDs20', 'STDs21'
'''

# cut head by id, due to lagging transformation
train_array = [df_train[df_train['id']==id].values[lag_window+40:,:] for id in df_train['id'].unique()]
train_array = np.concatenate(train_array).astype(np.float32)

# build train data matrix
train_X = train_array[:,4:]
train_y = train_array[:,3]

# split train data set into train and validation sub sets
total_count = train_array.shape[0]
val_count = int(train_array.shape[0]*0.2)

val_X = train_X[-1*val_count:,:]
val_y = train_y[-1*val_count:]
train_X = train_X[:total_count-val_count,:]
train_y = train_y[:total_count-val_count]

# build test data matrix
df_mean = test_df[lag_cols].rolling(window=lag_window).mean()
df_std = test_df[lag_cols].rolling(window=lag_window).std()
df_mean.columns = ['MA'+s for s in lag_cols]
df_std.columns = ['STD'+s for s in lag_cols]
df_test = pd.concat([test_df,df_mean,df_std], axis=1, join='inner')
# select last row
test_array = [df_test[df_test['id']==id].values[-1:,:] for id in df_test['id'].unique()]
test_array = np.concatenate(test_array).astype(np.float32)
# build the matrix
test_X = test_array[:,4:]
test_y = test_array[:,3]


'''
Traditional models
'''
# GBM
from sklearn.ensemble import GradientBoostingClassifier
model_GBM = GradientBoostingClassifier(random_state=42, verbose=1)
model_GBM.fit(train_X, train_y)
printScores(model_GBM.predict(train_X), train_y)
printScores(model_GBM.predict(val_X), val_y)
printScores(model_GBM.predict(test_X), test_y)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model_Linear = LogisticRegression(C=1, penalty='l1', tol=0.0001, max_iter=1000, verbose=1)
model_Linear.fit(train_X, train_y)
printScores(model_Linear.predict(train_X), train_y)
printScores(model_Linear.predict(val_X), val_y)
printScores(model_Linear.predict(test_X), test_y)
