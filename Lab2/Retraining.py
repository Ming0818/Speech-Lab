from proto2 import *
from prondict import *
import numpy as np
import matplotlib.pyplot as plt

if __name__== "__main__":
    data = np.load('data/lab2_data.npz')['data'][10]
    print(data)
    phoneHMMs = np.load('data/lab2_models.npz')['phoneHMMs'].item()

    prondict = prondict_list()

    modellist = {}
    for digit in prondict.keys():
        modellist[digit] = ['sil'] + prondict[digit] + ['sil']

    wordHMMs = {}
    wordHMMs = concatHMMs(phoneHMMs, modellist['4'])

    iteration_n = 30

    mean = wordHMMs['means']
    variance = wordHMMs['covars']
    loglik = -99999999
    for i in range(iteration_n):
        # 4.1 Gaussian emission probabilities
        result_obs = log_multivariate_normal_density_diag(data['lmfcc'], mean,variance)
        # 4.2 forward algorithm
        result_log = forward(result_obs, np.log(wordHMMs['startprob']), np.log(wordHMMs['transmat']))
        # 4.4 backward Function
        result_logbeta = backward(result_obs, np.log(wordHMMs['startprob']), np.log(wordHMMs['transmat']))
        # 5.1 State posterior probabilities
        result_loggamma = statePosteriors(result_log, result_logbeta)
        # 5.2 retrain
        mean, variance = updateMeanAndVar(data['lmfcc'], result_loggamma)

        # 4.2 forward likelihood P(X|θ) of the whole sequence
        N, M = result_log.shape
        loglik_before = loglik
        loglik = logsumexp(result_log[N - 1])
        print(loglik)
        if loglik - loglik_before <= 1.0:
            break
    print(np.exp(loglik))