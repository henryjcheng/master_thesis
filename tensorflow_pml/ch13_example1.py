"""
Implementation of the book Python Machine Learning Ch. 13 on Tensorflow
page 426
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# ========================================
# Warm-up Exercise:
#   Use simple scalars from Tensorflow to compute a net input z 
#   of a sample point x in a one-dimensional dataset with weight w and bias b
#       z = wx + b
# ========================================

## create a graph
g = tf.Graph()
with g.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(None), name='x')
    w = tf.Variable(2.0, name='weight')
    b = tf.Variable(0.7, name='bias')
    z = w*x + b

    init = tf.global_variables_initializer()

## create a session and pass in graph g
with tf.Session(graph=g) as sess:
    ## initialize w and b:
    sess.run(init)

    ## evaluate z
    print('\n')
    for t in [1.0, 0.6, -1.8]:
        print(f'x={t} --> z={sess.run(z, feed_dict={x:t})}')

## feed values to x as a batch
print('\n')
with tf.Session(graph=g) as sess:
    sess.run(init)
    print(sess.run(z, feed_dict={x: [1., 2., 3.]}))
