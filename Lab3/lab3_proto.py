import numpy as np
from lab3_tools import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=False):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    phoneTrans = []
    if addSilence:
        phoneTrans = ['sil']
    for i in range(0,len(wordList)):
        digit = wordList[i]
        phoneTrans = phoneTrans+ pronDict[digit]
        if addShortPause:
            phoneTrans = phoneTrans+['sp']
    if addSilence:
        phoneTrans = phoneTrans+['sil']

    return phoneTrans

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """

def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """

def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    new_H = {}
    sp_num_tran = hmmmodels['sp']['transmat'].shape[0]
    pho_num_tran = hmmmodels['sil']['transmat'].shape[0]

    sp_num = namelist.count('sp')
    all_num = len(namelist)
    pho_num = all_num-sp_num

    new_M = (pho_num_tran + 1) * pho_num + (sp_num_tran + 1) * sp_num - (all_num - 1) * 2
    new_D = pho_num_tran * pho_num + sp_num_tran * sp_num - (all_num - 1)
    new_trans = np.zeros((new_M, new_D))

    hmm = hmmmodels[namelist[0]]
    num_tran = hmm['transmat'].shape[0]
    new_trans[0:num_tran + 1, 0:num_tran] = np.vstack((hmm['startprob'], hmm['transmat']))
    new_means = hmm['means']
    new_covars = hmm['covars']

    start_M = num_tran+1-2
    start_D = num_tran-1

    for i in range(1,all_num):
        hmm = hmmmodels[namelist[i]]
        num_tran = hmm['transmat'].shape[0]

        temp = new_trans[start_M, start_D]
        new_trans[start_M:start_M+num_tran+1, start_D:start_D+num_tran] = np.vstack((hmm['startprob'], hmm['transmat']))
        new_trans[start_M, start_D:start_D + num_tran] = new_trans[start_M, start_D:start_D + num_tran].dot(temp)

        new_means = np.vstack((new_means, hmm['means']))
        new_covars = np.vstack((new_covars, hmm['covars']))
        start_M += num_tran+1-2
        start_D += num_tran-1
    new_H.update({'transmat':new_trans[1:new_trans.shape[0]]})
    new_H.update({'means': new_means})
    new_H.update({'covars': new_covars})
    new_H.update({'startprob': new_trans[0]})

    return new_H