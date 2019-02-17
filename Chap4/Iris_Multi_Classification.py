from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("../Dataset/iris.csv", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

sns.pairplot(df, hue='species')
plt.show()

dataset = df.values
X = dataset[:, 0:4].astype(float)
y_obj = dataset[:, 4]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)
y_encoded = np_utils.to_categorical(y)

model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y_encoded, epochs=50, batch_size=1)

print("\n Accuracy: %.4f" % (model.evaluate(X, y_encoded)[1]))