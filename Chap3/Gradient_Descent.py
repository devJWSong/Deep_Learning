import tensorflow as tf

data = [[2, 81], [4, 93], [6, 91], [8, 97]]
X_data = [X_row[0] for X_row in data]
y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

y = a * X_data + b

rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
learning_rate = 0.1

gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(gradient_decent)

        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, gradient a = %.4f, intercept b = %.4f" % (step, sess.run(rmse), sess.run(a), sess.run(b)))