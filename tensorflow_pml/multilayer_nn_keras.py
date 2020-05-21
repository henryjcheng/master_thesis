"""
Implementation of the book Python Machine Learning Ch. 13 on Tensorflow
page 438

Building multilayer perceptron to classify MNIST using Keras
"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras as keras
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

np.random.seed(123)
tf.set_random_seed(123)

## converts training labels into one-hot format
y_train_onehot = keras.utils.to_categorical(y_train)
print(f'First 3 labels: ', y_train[:3])
print(f'\nFirst 3 labels (one-hot):\n ', y_train_onehot[:3])

## Build Model
#   3 layrers, first 2 layers have 50 hidden units
#   tanh activation function
#   last layer has 10 layers for the 10 class labels
#       - use softmax to give prob of each class
model = keras.models.Sequential()

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=X_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'
    )
)

model.add(
    keras.layers.Dense(
        units=50,
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'
    )
)

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'
    )
)

sgd_optimizer = keras.optimizers.SGD(lr=.001, decay=1e-7, momentum=.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')

## Model Training
history = model.fit(X_train_centered, y_train_onehot,
                    batch_size=128, epochs=50,
                    verbose=1, validation_split=.1)

y_train_pred = model.predict_classes(X_train_centered, verbose=0)
print('First 3 predictions: ', y_train_pred[:3])

## Print Model Accuracy
y_train_pred = model.predict_classes(X_train_centered, verbose=0)
correct_preds = np.sum(y_train == y_train_pred, axis=0)
train_acc = correct_preds/y_train.shape[0]

print(f'Training Accuracy: {train_acc * 100}%')

y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds/y_test.shape[0]

print(f'Testing Accuracy: {test_acc * 100}%')