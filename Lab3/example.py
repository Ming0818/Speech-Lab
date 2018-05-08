import numpy as np
import matplotlib.pyplot as plt

from mfcc_lab1 import *
from prondict_lab2 import prondict
from lab3_proto import *
from lab3_tools import *

if __name__== "__main__":
    # phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    # phones = sorted(phoneHMMs.keys())
    # nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    # stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    # stateList
    # stateList.index('ay_2')

    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    wordTrans = list(path2info(filename)[2])
    phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=False)
    print(phoneTrans)