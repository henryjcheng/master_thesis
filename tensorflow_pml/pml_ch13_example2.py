"""
Implementation of the book Python Machine Learning Ch. 13 on Tensorflow
page 427

- create a rank 3 tensor of size batchsize x 2 x 3
- reshape 
- calculate column sums using Tensorflow's expressions

batchsize is unknown so we specify None for the batch size in the argument
for the shape parameter of the placeholder x
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32,
                       shape={None, 2, 3},
                       name='input_x')
    x2 = tf.reshape(x, shape=(-1, 6), name='x2')

    ## calculate the sum of each column
    xsum = tf.reduce_sum(x2, axis=0, name='col_sum')

    ## claculate the mean of each column
    xmean = tf.reduce_mean(x2, axis=0, name='col_mean')

with tf.Session(graph=g) as sess:
    x_array = np.arange(18).reshape(3, 2, 3)

    print('input shape: ', x_array.shape)
    print('Reshaped:\n', 
           sess.run(x2, feed_dict={x:x_array}))
    print('Column Sums:\n',
           sess.run(xsum, feed_dict={x:x_array}))
    print('Column Means:\n',
           sess.run(xmean, feed_dict={x:x_array}))
