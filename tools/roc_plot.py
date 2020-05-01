#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc(y, y_hat):
    plt.style.use('seaborn-darkgrid')
    lw = 2
    n_classes = 10
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(10):
      fpr[i], tpr[i], _ = roc_curve(y[:, i], y_hat[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(n_classes):
      plt.plot(fpr[i], tpr[i], lw=lw,
              label='Digit {} (area = {})'.format(i,format(roc_auc[i], '0.4f')))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight="bold")
    plt.ylabel('True Positive Rate', fontweight="bold")
    plt.title('ROC Curve', fontweight="bold")
    plt.legend(loc="lower right", prop={"weight":"bold"})
    plt.show()

def plot_pr(y, y_hat):
    plt.style.use('seaborn-darkgrid')
    lw = 2
    n_classes = 10
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(10):
       fpr[i], tpr[i], _ = precision_recall_curve(y[:, i], y_hat[:, i])

    for i in range(n_classes):
       plt.plot(fpr[i], tpr[i], lw=lw,
               label='Digit {}'.format(i))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight="bold")
    plt.ylabel('Precision', fontweight="bold")
    plt.title('Precision-Recall ', fontweight="bold")
    plt.legend(loc="lower right", prop={"weight":"bold"})
    plt.show()
