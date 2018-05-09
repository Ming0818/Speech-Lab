import numpy as np
import matplotlib.pyplot as plt

from lab1_mfcc import *
from lab2_prondict import prondict
from lab3_proto import *
from lab3_tools import *

import os
traindata = []
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

# for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
    for file in files:
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            samples, samplingrate = loadAudio(filename)

            lmfcc = mfcc(samples)
            mspecs = mspec(samples, samplingrate=samplingrate)

            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=False)
            targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

            traindata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspecs, 'targets': targets})
            print(filename)

# np.savez('traindata.npz', traindata=traindata)
np.savez('testdata.npz', testdata=traindata)