from keras.models import Sequential, load_model
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

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

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []

for train, test in skf.split(X, y):
    model = Sequential()
    model.add(Dense(24, input_dim=60, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X[train], y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(X[test], y[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy: " % n_fold, accuracy)
