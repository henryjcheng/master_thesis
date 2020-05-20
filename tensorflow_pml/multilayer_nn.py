"""
Implementation of the book Python Machine Learning Ch. 13 on Tensorflow
page 434

Building multilayer perceptron to classify MNIST
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from load_mnist import load_mnist

## loading the data
X_train, y_train = load_mnist('../../data/mnist', kind='train')
print(f'Rows: {X_train.shape[0]},    Columns: {X_train.shape[1]}')

X_test, y_test = load_mnist('../../data/mnist', kind='t10k')
print(f'Rows: {X_test.shape[0]},    Columns: {X_test.shape[1]}')

## mean centering and normalization
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_train, X_test

print(X_train_centered.shape, y_train.shape)
print(X_test_centered.shape, y_test.shape)

## Model Building
#   3 fully connected layers
#   use tanh activation
#   use softmax as output layer
n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)
    tf_x = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')
    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
    
    h1 = tf.layers.dense(inputs=tf_x, units=50, activation=tf.tanh, name='layer1')
    h2 = tf.layers.dense(inputs=h1, units=50, activation=tf.tanh, name='layer2')
    logits = tf.layers.dense(inputs=h2, units=10, activation=None, name='layer3')

    predictions = {
        'classes':tf.argmax(logits, axis=1, name='predicted_classes'),
        'probabilities':tf.nn.softmax(logits, name='softmax_tensor')
    }

## define cost function and optimizer:
with g.as_default():
    cost = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    train_op = optimizer.minimize(loss=cost)
    init_op = tf.global_variables_initializer()

## Create a batch data generator
def create_batch_generator(X, y, batch_size=128, shuffle=False):
    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack(X_copy, y_copy)
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)
    
    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])

## Create a session to launch the graph
sess = tf.Session(graph=g)
## run the variable initialization operator
sess.run(init_op)

## 50 epochs of training
for epoch in range(50):
    training_costs = []
    batch_generator = create_batch_generator(X_train_centered, y_train, batch_size=64)
    
    for batch_X, batch_y in batch_generator:
        ## prepare a dict to feed data to our network
        feed = {tf_x:batch_X, tf_y:batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_costs.append(batch_cost)
    print(f' -- Epoch {epoch+1}    Avg. Training Loss: {np.mean(training_costs)}')

## do prediction on the test set:
feed = {tf_x:X_test_centered}
y_pred = sess.run(predictions['classes'], feed_dict=feed)

test_accuracy = 100 * np.sum(y_pred == y_test)/y_test.shape[0]
print(f'\nTest Accuracy: {test_accuracy}')
