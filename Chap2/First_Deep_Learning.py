from keras.models import Sequential
from keras.layers import Dense

import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

Data_set = numpy.loadtxt("../Dataset/ThoraricSurgery.csv", delimiter=",")

X = Data_set[:, 0:17]
y = Data_set[:, 17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=10)

print("\n Accuracy: %.4f" % (model.evaluate(X, y)[1]))
