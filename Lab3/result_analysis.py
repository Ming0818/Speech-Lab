import numpy as np
import matplotlib.pyplot as plt
import itertools
import keras
from lab1_mfcc import *
from lab2_prondict import prondict
from lab3_proto import *
from lab3_tools import *
import os
from math import *
from sklearn.metrics import confusion_matrix

def frame_state_level(prediction, test_y):
    t =sum(prediction == test_y)
    n = test_y.shape[0]
    result = t/(n*1.0)
    return result


def frame_phoneme_level(prediction, test_y):
    p = np.floor(prediction/3)
    y = np.floor(test_y/3)
    confusion_lmfcc_phoneme = confusion_matrix(y, p)
    return frame_state_level(p, y),confusion_lmfcc_phoneme


def get_each_utter(predition,utter_len):
    utter =[]
    start = 0
    for i in range(len(utter_len)):
        end = start + utter_len[i]
        temp = predition[start:end]
        utter.append(temp)
        start = end
    return utter


def LevenshteinDistance(prediction,y):
    m = len(prediction)
    n = len(y)
    d = np.zeros((m+1,n+1))
    for i in range(1,m+1):
        d[i,0] = i
    for j in range(1,n+1):
        d[0,j] = j
    for j in range(1,n+1):
        for i in range(1,m+1):
            if prediction[i-1] == y[j-1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            #     min(deletion,insertion,substitution)
            d[i,j] = min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+substitutionCost)
    return d[m,n]


def edit_distance_state_level(utter,utter_y):
    prediction = [k for k, g in itertools.groupby(utter)]
    y = [k for k, g in itertools.groupby(utter_y)]
    error = LevenshteinDistance(prediction,y)
    return error/(len(y)*1.0)


def edit_distance_phoneme_level(utter, utter_y):
    prediction = np.floor(utter / 3)
    y = np.floor(utter_y / 3)
    return edit_distance_state_level(prediction, y)


if __name__== "__main__":
    test_y = np.load("data_stan_nody/testdata_stan.npz")['targets'].T[0]
    p_lmfcc = np.load("prediction/prediction_lmfcc_3.npz")['prediciton']
    p_mspec = np.load("prediction/prediction_mspec_3.npz")['prediciton']
    utter_len = np.load('utter_len.npz')['lmfcc']

    # plot the posterior of the first utter
    # y = keras.utils.to_categorical(test_y, 61)[0:utter_len[0],:]
    # p = p_lmfcc[0:utter_len[0],:]
    # plt.title("utter:oo7o")
    # plt.subplot(2, 1, 1)
    # plt.plot(p)
    # plt.subplot(2, 1, 2)
    # plt.plot(y)
    # plt.show()

    p_lmfcc = np.argmax(p_lmfcc,axis=1)
    p_mspec = np.argmax(p_mspec,axis=1)

    # lmfcc: 1. frame-by-frame at the state level && 2. frame-by-frame at the phoneme level
    # accuracy_lmfcc_state = frame_state_level(p_lmfcc,test_y)
    # confusion_lmfcc_state = confusion_matrix(test_y,p_lmfcc)
    # accuracy_lmfcc_phoneme,confusion_lmfcc_phoneme = frame_phoneme_level(p_lmfcc, test_y)
    # plt.subplot(1,2,1)
    # plt.pcolormesh(confusion_lmfcc_state.T,vmax = 50000)
    # plt.subplot(1,2,2)
    # plt.pcolormesh(confusion_lmfcc_phoneme.T, vmax = 150000)
    # plt.show()

    # mspec: 1. frame-by-frame at the state level && 2. frame-by-frame at the phoneme level
    accuracy_mspec_state = frame_state_level(p_mspec,test_y)
    confusion_mspec_state = confusion_matrix(test_y, p_mspec)
    accuracy_mspec_phoneme,confusion_mspec_phoneme= frame_phoneme_level(p_mspec, test_y)
    plt.subplot(1, 2, 1)
    plt.pcolormesh(confusion_mspec_state.T, vmax=50000)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(confusion_mspec_phoneme.T, vmax=150000)
    plt.show()

    # # get each utter
    # utter_lmfcc = get_each_utter(p_lmfcc,utter_len)
    # utter_mspec = get_each_utter(p_mspec,utter_len)
    # utter_y = get_each_utter(test_y, utter_len)
    #
    # # lmfcc: 3. edit distance at the state level && 4. edit distance at the phoneme level
    # utter_state_er_lmfcc = [edit_distance_state_level(utter_lmfcc[i],utter_y[i]) for i in range(len(utter_y))]
    # utter_phoneme_er_lmfcc = [edit_distance_phoneme_level(utter_lmfcc[i],utter_y[i]) for i in range(len(utter_y))]
    #
    # # mspec: 3. edit distance at the state level && 4. edit distance at the phoneme level
    # utter_state_er_mspec = [edit_distance_state_level(utter_mspec[i],utter_y[i]) for i in range(len(utter_y))]
    # utter_phoneme_er_mspec = [edit_distance_phoneme_level(utter_mspec[i],utter_y[i]) for i in range(len(utter_y))]
