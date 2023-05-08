# -*- coding: utf-8 -*-
"""DL_final_project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V2BxdpkDtKhP3c-BrY80m1elAf8uw1Xo
"""

from zipfile import ZipFile
with ZipFile('AUDIO_DATASET.zip', 'r') as zip:
  zip.extractall()
  print('Done')

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa
import librosa.display
import wave
import random as rn
import tensorflow as tf
#from keras.utils import to_categorical
import seaborn as sns
# %matplotlib inline

DATA_DIR = 'free-spoken-digit-dataset-master/recordings'

filenames=[]
# Shuffle
for flist in os.listdir(DATA_DIR):
    filenames.append(flist)
rn.shuffle(filenames)

from tensorflow.keras.utils import to_categorical
#from keras.utils import to_categorical
test_speaker = 'theo'
train_mfccs = []
train_y = []
test_mfccs = []
test_y = []
pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]))))
for i in range(len(filenames)):
        
        struct = filenames[i].split('_')
        digit = struct[0]
        speaker = struct[1]
        wav, sr = librosa.load(os.path.join(DATA_DIR , filenames[i]))
        mfcc = librosa.feature.mfcc(wav)
        padded_mfcc = pad2d(mfcc,40)
        if speaker == test_speaker:
            test_mfccs.append(padded_mfcc)
            test_y.append(digit)
        else:
            train_mfccs.append(padded_mfcc)
            train_y.append(digit)
            
train_mfccs = np.array(train_mfccs)
train_y = to_categorical(np.array(train_y))
test_mfccs = np.array(test_mfccs)
test_y = to_categorical(np.array(test_y))
train_X_ex = np.expand_dims(train_mfccs, -1)
test_X_ex = np.expand_dims(test_mfccs, -1)

train_mfccs.shape

ip = tf.keras.layers.Input(shape=train_X_ex[0].shape)
#layer 1
m = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), activation='relu')(ip)
m = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(m)
m= tf.keras.layers.BatchNormalization()(m)
#layer 2
m = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), activation='relu')(ip)
m = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(m)
m= tf.keras.layers.BatchNormalization()(m)
#layer 3
m = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
m = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(m)
m= tf.keras.layers.BatchNormalization()(m)
#layer 4
m = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), activation='relu')(ip)
m = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(m)
m= tf.keras.layers.BatchNormalization()(m)
m = tf.keras.layers.Dropout(0.2)(m)
m = tf.keras.layers.Flatten()(m)
m = tf.keras.layers.Dense(64, activation='relu')(m)
m = tf.keras.layers.Dense(32, activation='relu')(m)
op = tf.keras.layers.Dense(10, activation='softmax')(m)
model = tf.keras.Model(inputs=ip, outputs=op)
checkpoint_path = "cp.ckpt"
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,                                                     save_best_only=True,                                                 mode='max',                                                 monitor='val_accuracy',                                                 verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(train_X_ex,
          train_y,
          epochs=50,
          batch_size=32,
          validation_data=(test_X_ex, test_y),
          callbacks=[cp_callback])

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

y_pred= model.predict(test_X_ex)
y_p= np.argmax(y_pred, axis=1)
y_pred=np.argmax(test_y, axis=1)
y_pred=y_pred.astype(int)
y_t=np.argmax(test_y, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_t, y_p) 
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_mtx, 
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

from keras.models import Sequential
from keras.layers import SimpleRNN
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
model_RNN=Sequential()
model_RNN.add(SimpleRNN(256,input_shape=train_mfccs.shape, return_sequences=True))
#layer 1
model_RNN.add(SimpleRNN(128, return_sequences=True))
model_RNN.add(Dropout(0.2))
#layer 2
model_RNN.add(SimpleRNN(128, return_sequences=True))
model_RNN.add(Dropout(0.2))
#layer 3
model_RNN.add(SimpleRNN(64))
model_RNN.add(Dropout(0.2))
#layer 4
model_RNN.add(Dense(32,activation='relu'))
model_RNN.add(Dropout(0.2))
#layer 5
model_RNN.add(Dense(10,activation='softmax'))

model_RNN.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history1 = model_RNN.fit(train_mfccs,
          train_y,
          epochs=50,
          batch_size=32,
          validation_data=(test_mfccs, test_y),
          callbacks=[cp_callback])

plt.plot(history1.history['accuracy'], label='Train Accuracy')
plt.plot(history1.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

y_pred_rnn= model_RNN.predict(test_X_ex)
y_p_rnn= np.argmax(y_pred_rnn, axis=1)
y_pred_rnn=np.argmax(test_y, axis=1)
y_pred_rnn=y_pred_rnn.astype(int)
y_t_rnn=np.argmax(test_y, axis=1)
confusion_mtx_rnn = tf.math.confusion_matrix(y_t_rnn, y_p_rnn) 
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_mtx_rnn, 
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()