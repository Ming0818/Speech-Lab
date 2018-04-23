from proto2 import *
from tools2 import *
from prondict import *
from concatHMMs_list import *
import numpy as np

# --------------------------------------
# score 44 utterances in the data array with each of the 11 HMM models

if __name__== "__main__":
    data = np.load('data/lab2_data.npz')['data']
    phoneHMMs = np.load('data/lab2_models.npz')['phoneHMMs'].item()

    prondict = prondict_list()

    modellist = {}
    for digit in prondict.keys():
        modellist[digit] = ['sil'] + prondict[digit] + ['sil']

    wordHMMs = concatHMMs_list(phoneHMMs, modellist)

    utterances = []
    u_index = []
    # forward
    # for i in range(data.shape[0]):
    #     ut = []
    #     for digit in wordHMMs.keys():
    #         result_obs = log_multivariate_normal_density_diag(data[i]['lmfcc'], wordHMMs[digit]['means'], wordHMMs[digit]['covars'])
    #         result_loglik = forward(result_obs, np.log(wordHMMs[digit]['startprob']), np.log(wordHMMs[digit]['transmat']))
    #         N, M = result_loglik.shape
    #         re_loglik = logsumexp(result_loglik[N - 1])
    #         ut.append(re_loglik)
    #     utterances.append(ut)
    #     u_index.append(ut.index(max(ut)))
    # print(u_index)

    # viterbi
    for i in range(data.shape[0]):
        ut = []
        for digit in wordHMMs.keys():
            result_obs = log_multivariate_normal_density_diag(data[i]['lmfcc'], wordHMMs[digit]['means'], wordHMMs[digit]['covars'])
            result_loglik, path = viterbi(result_obs, np.log(wordHMMs[digit]['startprob']), np.log(wordHMMs[digit]['transmat']))
            ut.append(result_loglik)
        u_index.append(ut.index(max(ut)))
    print(u_index)
