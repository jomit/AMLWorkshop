import pandas as pd
import numpy as np
import keras
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, Activation
from keras.models import Model

from PM_Prep import train_df, test_df, feature_cols, printScores

'''
Build matrix
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


# generate input LSTM matrix for test
seq_array_test = [test_df[test_df['id']==id][feature_cols].values[-sequence_length:] 
                       for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
seq_array_test = np.asarray(seq_array_test).astype(np.float32)

# generate labels for test
y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
label_array_test = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
label_array_test = label_array_test.reshape(label_array_test.shape[0],1).astype(np.float32)


'''
Build network
'''
(sampleSize, seqLen, numFeatures) = seq_array.shape

input_layer = Input(shape=(seqLen, 1, numFeatures))
conv1_1 = Conv2D(25, (3, 1),
                 strides = (1,1),
                 padding='same',
                 kernel_initializer='normal',
                 name='conv1_1',
                 activation='relu')(input_layer)
conv1_2 = Conv2D(25, (3, 1),
                 strides = (1,1),
                 padding='same',
                 kernel_initializer='normal',
                 name='conv1_2',
                 activation='relu')(conv1_1)
pool1 = MaxPooling2D(name='pool1',
                     pool_size=(2, 1),
                     strides=(2, 1),
                     padding='same', )(conv1_2)

conv2_1 = Conv2D(50, (3, 1),
                 strides = (1,1),
                 padding='same',
                 kernel_initializer='normal',
                 name='conv2_1',
                 activation='relu')(pool1)
conv2_2 = Conv2D(50, (3, 1),
                 strides = (1,1),
                 padding='same',
                 kernel_initializer='normal',
                 name='conv2_2',
                 activation='relu')(conv2_1)
pool2 = MaxPooling2D(name='pool2',
                     pool_size=(2, 1),
                     strides=(2, 1),
                     padding='same', )(conv2_2)

conv3_1 = Conv2D(2, (3, 1),
                 strides = (1,1),
                 padding='same',
                 kernel_initializer='normal',
                 name='conv3_1',
                 activation='relu')(pool2)
conv3_2 = Conv2D(2, (3, 1),
                 strides = (1,1),
                 padding='same',
                 kernel_initializer='normal',
                 name='conv3_2',
                 activation='relu')(conv3_1)

pool3 = GlobalAveragePooling2D(name='pool3')(conv3_2)
class_conf = Activation('softmax',
                       name='class_conf')(pool3)

model2 = Model(inputs=input_layer, outputs=class_conf)

seq_array2 = seq_array.reshape(sampleSize, seqLen, 1, numFeatures)
label_array2 = keras.utils.np_utils.to_categorical(label_array)

'''
Train and evaluate
'''
base_lr = 3e-3
model2.compile(optimizer=keras.optimizers.Adam(lr=base_lr), # 'adam'
                loss='categorical_crossentropy',
                metrics=['accuracy'])
def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto'),
            keras.callbacks.LearningRateScheduler(schedule)]

model2.fit(seq_array2, label_array2, epochs=10, batch_size=200, validation_split=0.1, verbose=1, callbacks=callbacks)

# make predictions and compute confusion matrix
y_pred = model2.predict(seq_array2,verbose=1, batch_size=200)
y_pred2 = y_pred[:,0] <0.5
y_true = label_array
printScores(y_pred2, y_true)


seq_array_test2 = seq_array_test.reshape(seq_array_test.shape[0], seqLen, 1, numFeatures)

y_pred_test = model2.predict(seq_array_test2)
y_pred_test2 = y_pred_test[:,0] <0.5
y_true_test = label_array_test
printScores(y_pred_test2, y_true_test)
