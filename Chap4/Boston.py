from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../Dataset/housing.csv", delim_whitespace=True, header=None)

dataset = df.values
X = dataset[:, 0:13]
y = dataset[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=200, batch_size=10)

y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = y_test[i]
    prediction = y_prediction[i]
    print("Actual price: {:.3f}, Predicted price: {:.3f}".format(label, prediction))