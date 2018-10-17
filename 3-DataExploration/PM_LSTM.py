import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from PM_Prep import train_df, test_df, feature_cols, printScores

'''
LSTM
'''
# functions to generate LSTM matrix [?, 50, 25]
def gen_sequence(id_df, seq_length, seq_cols):
    # Only sequences that meet the window-length are considered
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# function to generate labels [?, 1]
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

sequence_length = 50

# generate LSTM matrix
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, feature_cols)) 
           for id in train_df['id'].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# generate labels
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['label1']) 
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)


# build LSTM network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(input_shape=(sequence_length, nb_features), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(seq_array, label_array, epochs=10, batch_size=200, validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])

# check performance on train data set
y_pred = model.predict_classes(seq_array,verbose=1, batch_size=200)
y_true = label_array
printScores(y_pred, y_true)

# generate input LSTM matrix for test
seq_array_test = [test_df[test_df['id']==id][feature_cols].values[-sequence_length:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
seq_array_test = np.asarray(seq_array_test).astype(np.float32)

# generate labels for test
y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
label_array_test = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test = label_array_test.reshape(label_array_test.shape[0],1).astype(np.float32)

# check performance on test data set
y_pred_test = model.predict_classes(seq_array_test)
y_true_test = label_array_test
printScores(y_pred_test, y_true_test)
