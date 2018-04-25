from proto2 import *
from prondict import *
import numpy as np
import matplotlib.pyplot as plt

if __name__== "__main__":
    data = np.load('data/lab2_data.npz')['data']
    phoneHMMs = np.load('data/lab2_models.npz')['phoneHMMs'].item()
    list(sorted(phoneHMMs.keys()))
    phoneHMMs['ah'].keys()

    example = np.load('data/lab2_example.npz')['example'].item()
    list(example.keys())

    prondict = prondict_list()

    modellist = {}
    for digit in prondict.keys():
        modellist[digit] = ['sil'] + prondict[digit] + ['sil']

    wordHMMs = {}
    wordHMMs['o'] = concatHMMs(phoneHMMs, modellist['o'])

    imageN = 5

    # 4.1 Gaussian emission probabilities
    verify_obs = example['obsloglik']
    result_obs = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs['o']['means'], wordHMMs['o']['covars'])
    print((abs(verify_obs - result_obs) < 0.0000001).all())

    plt.subplot(imageN, 1, 1)
    plt.title("obsloglik", fontsize=10)
    plt.xticks([])
    plt.pcolormesh(result_obs.T)

    # 4.2 forward algorithm
    verify_log = example['logalpha']
    result_log = forward(result_obs, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    # print((abs(verify_log == result_log) < 0.0000001).all())
    print((verify_log == result_log).all())

    result_logalpha = result_log.copy()
    result_log[np.isneginf(result_log)] = 0
    plt.subplot(imageN, 1, 2)
    plt.title("forward algorithm", fontsize=10)
    plt.xticks([])
    plt.pcolormesh(result_log.T)

    # 4.2 forward likelihood P(X|θ) of the whole sequence
    N,M = result_log.shape
    result_loglik = logsumexp(result_log[N-1])
    verify_loglik = example['loglik']
    print(abs(verify_loglik - result_loglik)<0.0000001)

    # 4.3 Viterbi Approximation
    verify_vlog = example['vloglik']
    result_vlog, result_path = viterbi(result_obs, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    print(verify_vlog[0] == result_vlog)
    print((verify_vlog[1] == result_path).all())

    plt.subplot(imageN, 1, 3)
    plt.title("viterbi algorithm", fontsize=10)
    plt.xticks([])
    plt.pcolormesh(result_log.T)
    plt.plot(result_path, color = 'white')

    # 4.4 backward Function
    verify_logbeta = example['logbeta']
    result_logbeta = backward(result_obs, np.log(wordHMMs['o']['startprob']), np.log(wordHMMs['o']['transmat']))
    print((abs(verify_logbeta - result_logbeta)<0.0000001).all())

    result_logbeta1 = result_logbeta.copy()
    result_logbeta1[np.isneginf(result_logbeta1)] = 0
    plt.subplot(imageN, 1, 4)
    plt.title("backward algorithm", fontsize=10)
    plt.xticks([])
    plt.pcolormesh(result_logbeta1.T)

    # 4.4 backward likelihood P(X|θ) of the whole sequence
    N, M = result_logbeta.shape
    result_logbetalik = logsumexp(result_logbeta[0] + np.log(wordHMMs['o']['startprob']) + result_obs[0])
    verify_loglik = example['loglik']
    print(abs(verify_loglik - result_logbetalik) < 0.0000001)

    # 5.1 State posterior probabilities
    verify_loggamma = example['loggamma']
    result_loggamma = statePosteriors(verify_log, verify_logbeta)
    print((verify_loggamma == result_loggamma).all())

    result_loggamma[np.isneginf(result_loggamma)] = 0
    plt.subplot(imageN, 1, 5)
    plt.title("state posterior probabilities", fontsize=10)
    plt.xticks([])
    plt.pcolormesh(result_loggamma.T)
    plt.show()
    
    # 5.1.2 test state probabilities in linear domain
    sum_gamma2 = np.exp(result_loggamma)
    sum_gamma3 = np.sum(sum_gamma2,axis=1)
    print(sum_gamma3)
    
    # 5.2 retrain
    mean, variance = updateMeanAndVar(example['lmfcc'],result_loggamma)
