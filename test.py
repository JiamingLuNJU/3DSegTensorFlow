
import tensorflow as tf

mySess = tf.Session()

W = tf.Variable(tf.random_normal([20, 10], mean=20/10, stddev=0.3))

mySess.run(tf.global_variables_initializer())

print(mySess.run(W))