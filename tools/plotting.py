#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint

#============================================================================

def plot_history(history, title):
  plt.style.use('seaborn-darkgrid')
  plt.plot(history.history['val_accuracy'], marker='', linewidth=4, alpha=0.7)
  plt.plot(history.history['accuracy'], marker='', linewidth=4, alpha=0.7)
  plt.title('Model accuracy ' + title, fontweight="bold")
  plt.ylabel('Accuracy', fontweight="bold")
  plt.xlabel('epoch', fontweight="bold")
  plt.legend(['train_acc', 'test_acc'], loc='upper left', prop={"weight":"bold"})
  plt.show()
  plt.plot(history.history['loss'], marker='', linewidth=4, alpha=0.7)
  plt.plot(history.history['val_loss'], marker='', linewidth=4, alpha=0.7)
  plt.title('Model Loss ' + title, fontweight="bold")
  plt.ylabel('Loss', fontweight="bold")
  plt.xlabel('epoch', fontweight="bold")
  plt.legend(['train_loss', 'test_loss'], loc='upper left', prop={"weight":"bold"})
  plt.show()
#===========================================================================
def barplot(train, test, title, color, index):
    # style
    if (randint(0,1)):plt.style.use('seaborn-whitegrid')
    else:plt.style.use('seaborn-darkgrid')

    df = pd.DataFrame({'Train': train, 'Validation': test}, index=index)
    # simple bar or barh
    bar = randint(0,1)
    if bar == 1:ax = df.plot.barh(rot = 0, color=color, fontsize=13)
    else: ax = df.plot.bar(rot = 0, color=color, fontsize=13)

    ax.set_alpha(0.8)
    if bar==1: ax.set_yticklabels(index, fontweight='bold')
    else: ax.set_xticklabels(index, fontweight='bold')
    if title=='Accuracy':
        if bar==0:ax.set_yticks([0, 0.5, 1, 1.5])
    for p in ax.patches:
        if bar == 1:
            left, bottom, width, height = p.get_bbox().bounds
            if title=='Accuracy':
                ax.annotate(str(format(width * 100, '.2f')) + "%", xy=(left+width/2, bottom+height/2),
                                ha='center', va='center', fontweight='bold')
            else:
                ax.annotate(str(np.round(width, decimals=3)), xy=(left+width/2, bottom+height/2),
                                ha='center', va='center', fontweight='bold')
        else:
            ax.annotate(np.round(p.get_height(),decimals=2),
                        (p.get_x()+p.get_width()/2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10),
                        textcoords='offset points', fontweight='bold')
    ax.set_title(title, fontweight="bold")
    plt.show()
def plotting(pred, T, y, index, history, best):

    if (randint(0,1)):plt.style.use('seaborn-whitegrid')
    else: plt.style.use('seaborn-darkgrid')

    for i in range(len(index)):
        if i==best:
            plt.plot(history[i].history[pred], marker='', linewidth=4, alpha=0.7)
        else:
            plt.plot(history[i].history[pred])
    plt.title(T + ' ' + y, fontweight="bold")
    plt.ylabel(y, fontweight="bold")
    plt.xlabel('epoch', fontweight="bold")
    plt.legend(index, loc='upper left', prop={"weight":"bold"})
    plt.show()
def get(model, history):
  layers = []
  for i in history:
    layers = layers + [mean(i.history[model])]
  return layers

def mean(list_val):
    return sum(list_val)/len(list_val)
#============================================================================

def barplot_acc_loss(index, history):
    # plotting average of each size
    barplot(get('accuracy', history), get('val_accuracy', history), 'Accuracy', ['g', "coral"], index)
    barplot(get('loss', history), get('val_loss', history), 'Loss', ['r', 'y'], index)

def plot_train_test(index, history, best):
    # plotting the progression of each size during epochs
    plotting('accuracy', 'Train', 'Accuracy', index, history, best)
    plotting('val_accuracy', 'Test', 'Accuracy', index, history, best)
    plotting('loss', 'Train', 'Loss', index, history, best)
    plotting('val_loss', 'Test', 'Loss', index, history, best)

def show_results(index, history):
    # results
    print("\nTraining accuracy : ")
    for i in range(len(index)):
        print(index[i]," : ", get('accuracy', history)[i])
    print("\nTest accuracy : ")
    for i in range(len(index)):
        print(index[i]," : ", get('val_accuracy', history)[i])
    print("\nTraining loss : ")
    for i in range((len(index))):
        print(index[i]," : ", get('loss', history)[i])
    print("\nTest loss : ")
    for i in range(len(index)):
        print(index[i]," : ", get('val_loss', history)[i])
