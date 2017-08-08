
import tensorflow as tf

mySess = tf.Session()

W = tf.Variable(tf.random_normal([20, 10], mean=1/10, stddev=1/20))

mySess.run(tf.global_variables_initializer())

print(mySess.run(W))