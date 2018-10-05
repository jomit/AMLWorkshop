'''
Predictive Maintainance v2
'''
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score


def printScores(y_pred, y_true):
    print()
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print( 'precision = ', precision, ', recall = ', recall)


'''
Data Exploration and Preparation
'''
CMAPSS = [['PM_train.txt', 'PM_test.txt', 'PM_truth.txt'],
            ['CMAPSSData/train_FD001.txt', 'CMAPSSData/test_FD001.txt', 'CMAPSSData/RUL_FD001.txt'],
            ['CMAPSSData/train_FD002.txt', 'CMAPSSData/test_FD002.txt', 'CMAPSSData/RUL_FD002.txt'],
            ['CMAPSSData/train_FD003.txt', 'CMAPSSData/test_FD003.txt', 'CMAPSSData/RUL_FD003.txt'],
            ['CMAPSSData/train_FD004.txt', 'CMAPSSData/test_FD004.txt', 'CMAPSSData/RUL_FD004.txt']]
dataFiles = CMAPSS[0]
dataColumns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read data 
train_df = pd.read_csv(dataFiles[0], sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = dataColumns

test_df = pd.read_csv(dataFiles[1], sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = dataColumns

rul_df = pd.read_csv(dataFiles[2], sep=" ", header=None)
rul_df.drop(rul_df.columns[[1]], axis=1, inplace=True)
rul_df.columns = ['more']
rul_df['id'] = rul_df.index + 1

# train set, calculate RUL
train_df = train_df.sort_values(['id','cycle'])
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

# test set, use ground truth to calculate RUL
test_df = test_df.sort_values(['id','cycle'])
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
rul_df['max'] = rul['max'] + rul_df['more']
rul_df.drop('more', axis=1, inplace=True)
test_df = test_df.merge(rul_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# label data
w1 = 30
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )

# normalize train data
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(['id','cycle','RUL','label1'])   # feature columns
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

# normalize test data
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)

# describe data and use only some columns
def describe():
    print('train set', train_df.shape)
    print('test set', test_df.shape)
    print('check distribution \n', train_df['label1'].value_counts())
    stats = train_df.describe().T
    unchanging_cols = list(stats[stats['std']==0].index)
    print('unchanging cols', unchanging_cols)
    # ['setting3', 's1', 's5', 's10', 's16', 's18', 's19']

print('Describe data:')
describe()
    
feature_cols = ['cycle_norm', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
feature_cols = [s for s in feature_cols if s not in ['setting3', 's1', 's5', 's10', 's16', 's18', 's19']]
    
cols = ['id','cycle','RUL','label1'] + feature_cols    
train_df = train_df[cols]
test_df = test_df[cols]
