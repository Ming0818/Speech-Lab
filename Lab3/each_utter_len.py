import numpy as np
import matplotlib.pyplot as plt

from lab1_mfcc import *
from lab2_prondict import prondict
from lab3_proto import *
from lab3_tools import *
import os

lmfcc_len =[]
mspec_len = []

for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
    for file in files:
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            samples, samplingrate = loadAudio(filename)

            lmfcc = mfcc(samples)
            mspecs = mspec(samples, samplingrate=samplingrate)

            lmfcc_len.append(lmfcc.shape[0])
            mspec_len.append(mspecs.shape[0])


np.savez('utter_len.npz', lmfcc = lmfcc_len,mspec = mspec_len)