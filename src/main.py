#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import sklearn

sys.path.insert(0, '..')

# Importing packages
from tools.image_preprocessing import *
from tools.plotting import *
from tools.history import *
from tools.roc_plot import *
from tools.learning_rate import *

# ==============================================================================
# Scaled & well prepared Data
(train_x, train_y), (test_x, test_y) = preparing_data()

# ============================================================================
# load the model
from keras.models import load_model

classifier = load_model("my_model.h5py")

# predict test_y
Y_pred = classifier.predict(test_x)
# ============================================================================
# Classification Report
print('\n', sklearn.metrics.classification_report(np.where(test_y > 0)[1],
                                                  np.argmax(Y_pred, axis=1),
                                                  target_names=list(dict_characters.values())),
      sep='')

# ==============================================================================
# decode (one hot coding)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(test_y, axis=1)

# Confusion Matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=list(dict_characters.values()))
# ==========================================================================

# plot the roc curves
plot_roc(test_y, Y_pred)

# plot precisions & recalls
plot_pr(test_y, Y_pred)
# ==============================================================================
