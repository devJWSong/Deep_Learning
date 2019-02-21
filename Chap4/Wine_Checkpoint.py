from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import  ModelCheckpoint

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('../Dataset/wine.csv', header=None)
df = df_pre.sample(frac=1)
dataset = df.values
X = dataset[:, 0:12]
y = dataset[:, 12]

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

MODEL_DIR = './model'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1, save_best_only=True)

model.fit(X, y, validation_split=0.2, epochs=200, batch_size=200, verbose=0, callbacks=[checkpointer])

print("\n Accuracy: %.4f" % (model.evaluate(X, y)[1]))
