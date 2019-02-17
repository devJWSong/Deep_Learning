from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../Dataset/sonar.csv", header=None)

dataset = df.values
X = dataset[:, 0:60]
y_obj = dataset[:, 60]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=130, batch_size=5)

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))