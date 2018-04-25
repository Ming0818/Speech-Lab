from tools2 import *

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
    M = hmmmodels['sil']['means'].shape[0]
    D = hmmmodels['sil']['means'].shape[1]
    num_tran = hmmmodels['sil']['transmat'].shape[0]

    # 数量
    pho_num = len(namelist)

    new_trans = np.zeros((pho_num *(M+1)-2,pho_num *(M+1)-2))
    new_means = np.zeros((M * pho_num, D))
    new_covars = np.zeros(new_means.shape)
    new_startprob = np.zeros(pho_num* M)
    new_startprob[0] = 1

    for i in range(pho_num):
        hmm = hmmmodels[namelist[i]]
        new_trans[i*M:i*M+num_tran, i*M:i*M+num_tran] = hmm['transmat']
        new_means[i*M:(i+1)*M] = hmm['means']
        new_covars[i*M:(i+1)*M] = hmm['covars']
    new_H.update({'transmat':new_trans})
    new_H.update({'means': new_means})
    new_H.update({'covars': new_covars})
    new_H.update({'startprob': new_startprob})

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
    N, M = log_emlik.shape  # 71, 9
    logalpha = np.zeros(log_emlik.shape)
    for n in range(N):
        for j in range(M):
            if n == 0:
                logalpha[n, j] = log_startprob[j] + log_emlik[n, j]
            else:
                logalpha[n, j] = logsumexp(logalpha[n-1, :]+log_transmat[0:M, j]) + log_emlik[n, j]
    return logalpha


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape  # 71, 9
    logbeta = np.zeros(log_emlik.shape)
    for n in range(N-2, -1, -1):
        for i in range(M):
            logbeta[n, i] = logsumexp(log_transmat[i, 0:M] + log_emlik[n+1, :] + logbeta[n+1, :])
    return logbeta

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
    viterbi_path = np.zeros([N],dtype=int)
    V = np.zeros([N,M])
    B = np.zeros([N,M])

    for i in range(M):
        V[0,i] = log_startprob[i] + log_emlik[0,i]

    for t in range(1,N):
        for j in range(M):
            V[t,j] = np.max(V[t-1,:] + log_transmat[0:M, j]) + log_emlik[t,j]
            B[t,j] = np.argmax(V[t-1,:] + log_transmat[0:M, j])

    viterbi_loglik = np.max(V[N-1,:])
    viterbi_path[N-1] = np.argmax( V[N-1,:])

    for t in range(N-2, -1, -1):
        viterbi_path[t] = B[t+1, viterbi_path[t+1]]
    
    return viterbi_loglik, viterbi_path

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    N, M = log_alpha.shape
    log_gamma = np.zeros([N,M])
    for n in range(N):
        for i in range(M):
            log_gamma[n, i] = log_alpha[n, i] + log_beta[n, i] - logsumexp(log_alpha[N-1])
    return log_gamma

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
    N,D = X.shape #D = 13
    N,M = log_gamma.shape #M = 9
    means = np.zeros([M,D])
    covars = np.zeros([M,D])

    # a:分子,b:分母,m:means,c：covariance
    for i in range(M):
        m_a = 0
        m_b = 0
        c_a = 0
        c_b = 0
        for t in range(N):
            m_a =m_a + np.exp(log_gamma[t,i]) * X[t,:]
            m_b = m_b + np.exp(log_gamma[t,i])
        means[i,:] = m_a / m_b

        for t in range(N):
            # for d in range(D):
            c_a = c_a + np.exp(log_gamma[t,i])* (X[t,:])**2
            c_b = c_b + np.exp(log_gamma[t,i])
        covars[i,:] = c_a /c_b - means[i,:]**2

    covars[covars<varianceFloor] = varianceFloor
    return means, covars
