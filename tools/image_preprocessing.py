#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

def preparing_data():
    # import data
    (train_images, train_y) ,(test_images, test_y) = mnist.load_data()

    # convert pixels from int to float
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    # normalizing data
    train_x = (train_images/255)
    test_x = (test_images/255)

    # Encode dependent variable to categorical
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # Reshape sets to [observations][width][height][image_version]
    train_x = train_x.reshape(train_x.shape[0],28,28,1)
    test_x = test_x.reshape(test_x.shape[0],28,28,1)

    return ((train_x, train_y), (test_x, test_y))

    
