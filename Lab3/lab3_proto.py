import numpy as np
from lab3_tools import *
from tools import *

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
    example = np.load('lab3_example.npz')['example'].item()

    # -----------get the whole HMM---------------
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    # print((utteranceHMM['transmat'] == example['utteranceHMM']['transmat']).all())

    # -----------viterbi results-----------
    result_obs = log_multivariate_normal_density_diag(lmfcc, utteranceHMM['means'], utteranceHMM['covars'])
    result_vlog, result_path = viterbi(result_obs, np.log(utteranceHMM['startprob']), np.log(utteranceHMM['transmat']))
    # print((result_path==example['viterbiPath']).all())
    # print(result_vlog==example['viterbiLoglik'])

    # state trans
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]

    # -----------Use stateTrans to convert the sequence of Viterbi states (corresponding to
    # -----------the utteranceHMM model) to the unique state names in stateList.
    viterbiStateTrans = []
    for i in range(len(result_path)):
        viterbiStateTrans.append(stateTrans[result_path[i]])
    # print((viterbiStateTrans == example['viterbiStateTrans']))

    return viterbiStateTrans

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


def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape  # N frames, M states
    viterbi_path = np.zeros([N], dtype=int)
    V = np.zeros([N, M])
    B = np.zeros([N, M])

    for i in range(M):
        V[0, i] = log_startprob[i] + log_emlik[0, i]

    for t in range(1, N):
        for j in range(M):
            V[t, j] = np.max(V[t - 1, :] + log_transmat[0:M, j]) + log_emlik[t, j]
            B[t, j] = np.argmax(V[t - 1, :] + log_transmat[0:M, j])

    viterbi_loglik = np.max(V[N - 1, :])
    viterbi_path[N - 1] = np.argmax(V[N - 1, :])

    for t in range(N - 2, -1, -1):
        viterbi_path[t] = B[t + 1, viterbi_path[t + 1]]

    return viterbi_loglik, viterbi_path
