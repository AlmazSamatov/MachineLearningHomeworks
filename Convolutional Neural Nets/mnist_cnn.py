import keras
import tensorflow as tf
# Deep learning library. Tensors are just multi-dimensional arrays

## Loading MNIST Dataset from Keras split it into Training and test
from keras.backend import sparse_categorical_crossentropy
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Conv2D
from keras.optimizers import adagrad, sgd

mnist = keras.datasets.mnist
# mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  
# unpacks images to x_train/x_test and labels to y_train/y_test

## Normalize Data
x_train = keras.utils.normalize(x_train, axis=1)
# scales data between 0 and 1
x_test = keras.utils.normalize(x_test, axis=1)
# scales data between 0 and 1

#x_train = x_train.reshape(60000,28,28,1)
#x_test = x_test.reshape(10000,28,28,1)

## Build a CNN Model
model = keras.models.Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# a basic feed-forward model
model.add(keras.layers.Flatten())
# takes our 28x28 and makes it 1x784

model.add(keras.layers.Dense(784, activation=tf.nn.relu))
# a simple fully-connected layer, 128 units, relu activation

model.add(keras.layers.Dense(256, activation=tf.nn.relu))

# a simple fully-connected layer, 128 units, relu activation
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
# our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

## Train the Model
model.fit(x_train, y_train, epochs=30, batch_size=100)

## Evaluate the Model
val_loss, val_acc = model.evaluate(x_test, y_test)  
# evaluate the out of sample data with model
print(val_loss)  
# model's loss (error)
print(val_acc)  
# model's accuracy


## Predict
predictions = model.predict(x_test)
print(predictions)


## Show the Predicted image
import numpy as np
import matplotlib.pyplot as plt
print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
