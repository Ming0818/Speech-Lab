import numpy as np
import matplotlib.pyplot as plt

from lab1_mfcc import *
from lab2_prondict import prondict
from lab3_proto import *
from lab3_tools import *

if __name__== "__main__":
    # phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    # phones = sorted(phoneHMMs.keys())
    # nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    # stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    # stateList
    # stateList.index('ay_2')

    # filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    filename = '/afs/kth.se/misc/csc/dept/tmh/corpora/tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    example = np.load('lab3_example.npz')['example'].item()
    list(example.keys())
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    wordTrans = list(path2info(filename)[2])
    phoneTrans = words2phones(wordTrans, prondict, addSilence=True, addShortPause=False)
    print(phoneTrans)

    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    print((utteranceHMM['transmat'] == example['utteranceHMM']['transmat']).all())

    # state trans
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]

    # viterbi results
    result_obs = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    result_vlog, result_path = viterbi(result_obs, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))

    print((result_path==example['viterbiPath']).all())
    print(result_vlog==example['viterbiLoglik'])

    # Use stateTrans to convert the sequence of Viterbi states (corresponding to
    # the utteranceHMM model) to the unique state names in stateList.
    stateList = []
    for i in range(len(result_path)):
        stateList.append(stateTrans[result_path[i]])
    print((stateList==example['viterbiStateTrans']))