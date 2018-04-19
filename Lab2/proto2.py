import numpy as np
from tools2 import *
from prondict import *

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
    # M = 3, D = 13
    M = phoneHMMs['sil']['means'].shape[0]
    D = phoneHMMs['sil']['means'].shape[1]
    num_tran = phoneHMMs['sil']['transmat'].shape[0]

    # 数量
    pho_num = len(namelist)

    new_trans = np.zeros((pho_num *(M+1)-2,pho_num *(M+1)-2))
    new_means = np.zeros((M * pho_num, D))
    new_covars = np.zeros(new_means.shape)

    for i in range(pho_num):
        hmm = phoneHMMs[namelist[i]]
        new_trans[i*M:i*M+num_tran, i*M:i*M+num_tran] = hmm['transmat']
        new_means[i*M:(i+1)*M] = hmm['means']
        new_covars[i*M:(i+1)*M] = hmm['covars']
    new_H.update({'transmat':new_trans})
    new_H.update({'means': new_means})
    new_H.update({'covars':new_covars})

    return new_H


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

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

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """


if __name__== "__main__":
    data = np.load('lab2_data.npz')['data']
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    list(sorted(phoneHMMs.keys()))
    phoneHMMs['ah'].keys()

    example = np.load('lab2_example.npz')['example'].item()
    list(example.keys())

    prondict = prondict_list()

    modellist = {}
    for digit in prondict.keys():
        modellist[digit] = ['sil'] + prondict[digit] + ['sil']

    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, modellist['o'])
