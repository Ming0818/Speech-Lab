import numpy as np
import matplotlib.pyplot as plt

from lab1_mfcc import *
from lab2_prondict import prondict
from lab3_proto import *
from lab3_tools import *
import os

MAN_SAMPLES = 4290
WOMAN_SAMPLES = 4445

traindata = []
valdata = []
testdata = []
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()

man_count, woman_count = 0, 0
man, woman = [], []

# for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
for root, dirs, files in os.walk('/afs/kth.se/misc/csc/dept/tmh/corpora/tidigits/disc_4.1.1/tidigits/train'):
    for file in files:
        if file.endswith('.wav'):
            filename = os.path.join(root, file)
            samples, samplingrate = loadAudio(filename)

            lmfcc = mfcc(samples)
            mspecs = mspec(samples, samplingrate=samplingrate)

            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=False)
            targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

            if root.__contains__("woman") and woman_count < 6:
                valdata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspecs, 'targets': targets})
                if not woman.__contains__(root[-2:]):
                    woman_count += 1
                    woman.append(root[-2:])
            elif not root.__contains__("woman") and man_count < 6:
                valdata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspecs, 'targets': targets})
                if not man.__contains__(root[-2:]):
                    man_count += 1
                    man.append(root[-2:])
            else:
                traindata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspecs, 'targets': targets})
            print(filename)

np.savez('/tmp/speech0511/traindata.npz', traindata=traindata)
np.savez('/tmp/speech0511/valdata.npz', valdata=valdata)

# for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
# for root, dirs, files in os.walk('/afs/kth.se/misc/csc/dept/tmh/corpora/tidigits/disc_4.1.1/tidigits/test'):
#     for file in files:
#         if file.endswith('.wav'):
#             filename = os.path.join(root, file)
#             samples, samplingrate = loadAudio(filename)
#
#             lmfcc = mfcc(samples)
#             mspecs = mspec(samples, samplingrate=samplingrate)
#
#             wordTrans = list(path2info(filename)[2])
#             phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=False)
#             targets = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)
#
#             testdata.append({'filename': filename, 'lmfcc': lmfcc,'mspec': mspecs, 'targets': targets})
#             print(filename)
#
# np.savez('testdata.npz', testdata=testdata)