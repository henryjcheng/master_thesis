"""
Implementation of the book Python Machine Learning Ch. 13 on Tensorflow
page 428

Implement OLS regression

- implement class TfLinreg
- define trainable variable w, b
- define model z = w*x + b
- define cost function to be Mean Squared Error (MSE)
- use gradient descent optimizer to learn the weight
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

x_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0,
                    6.3, 6.6, 7.4, 8.0, 9.0])

class TfLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01, random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()

        ## build the model
        with self.g.as_default():
            ## set graph-level random-seed
            tf.set_random_seed(random_seed)

            self.build()
            ## create initializer
            self.init_op = tf.global_variables_initializer()
    
    def build(self):
        ## define placeholders for inputs
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.x_dim), name='x_input')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None), name='y_input')
        print(self.x)
        print(self.y)

        ## define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape=(1)), name='weight')
        b = tf.Variable(tf.zeros(shape=(1)), name='bias')
        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.x + b, name='z_net')
        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net, name='sqr_errors')
        print(sqr_errors)
        self.mean_cost = tf.reduce_mean(sqr_errors, name='mean_cost')

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate = self.learning_rate,
            name='GradientDescent'
        )
        self.optimizer = optimizer.minimize(self.mean_cost)


## Training
##  - implement separate function that needs the following as inputs:
#       * TensorFlow session
#       * model instance
#       * training data
#       * Number of epochs
##  - this function performs:
#       1. initialize variables in the tf session using init_op
#       2. iterate and call optimizer while feeding training data
def train_linreg(sess, model, x_train, y_train, num_epochs=10):
    ## initialize all variables: W and b
    sess.run(model.init_op)

    training_costs = []
    for i in range(num_epochs):
        _, cost = sess.run([model.optimizer, model.mean_cost], feed_dict={model.x:x_train, model.y:y_train})
        training_costs.append(cost)

    return training_costs

lrmodel = TfLinreg(x_dim=x_train.shape[1], learning_rate=0.01)
sess = tf.Session(graph=lrmodel.g)
training_costs = train_linreg(sess, lrmodel, x_train, y_train)

## Visualization
plt.plot(range(1, len(training_costs) + 1), training_costs)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Training Cost')
plt.savefig('training_cost.png')
plt.clf()

## Prediction
def predict_linreg(sess, model, x_test):
    y_pred = sess.run(model.z_net, feed_dict={model.x:x_test})
    return y_pred

plt.scatter(x_train, y_train, marker='s', s=50, label='Training Data')
plt.plot(range(x_train.shape[0]), predict_linreg(sess, lrmodel, x_train),
         color='gray', marker='o',
         markersize=6, linewidth=3,
         label='LinReg Model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('prediction.png')
plt.clf()