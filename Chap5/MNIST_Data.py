from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np
import sys
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

(X_train, y_class_train), (X_test, y_class_test) = mnist.load_data()

print("Num of images in Train set: %d" % (X_train.shape[0]))
print("Num of images in Test set: %d" % (X_test.shape[0]))

import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap='Greys')
plt.show()

for x in X_train[0]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

X_train = X_train.reshape(X_train.shape[0], 784)
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64') / 255

print('class: %d ' % (y_class_train[0]))

y_train = np_utils.to_categorical(y_class_train, 10)
y_test = np_utils.to_categorical(y_class_test, 10)

print(y_train[0])