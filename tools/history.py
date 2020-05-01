#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers import Flatten, LeakyReLU
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import RMSprop, Adagrad, Adam, Nadam, SGD, Adadelta
from keras.preprocessing.image import ImageDataGenerator
#==============================================================================

def get_history(nb_classifiers, classifiers, train_x, train_y, batch, epochs):
    history = [0] * nb_classifiers
    i = 0
    for classifier in classifiers:
      history[i] = classifier.fit(train_x, train_y, validation_split=0.25, batch_size = batch, epochs = epochs)
      i += 1
    return history

def get_history_with_test(nb_classifiers, classifiers, train, test, batch, epochs):
    history = [0] * nb_classifiers
    i = 0
    for classifier in classifiers:
      history[i] = classifier.fit(train[0], train[1],
                                  validation_data = (test[0], test[1]),
                                  batch_size = batch, epochs = epochs)
      i += 1
    return history
