import numpy as np
from sklearn.mixture import *
import proto
import matplotlib.pyplot as plt
from tools import *

np.random.seed(10)


def Get_Most(list):
    item_num = dict((item,list.count(item)) for item in list)
    i = max(item_num.items(), key= lambda x:x[1])[0]
    return i

if __name__== "__main__":
    data = np.load('lab1_data.npz')['data']
    mfcc_length = []
    index = []
    piece = data[0]
    samples = piece['samples']
    samples_samplingrate = piece['samplingrate']
    # mfcc
    mfccs = proto.mfcc(samples, samplingrate=samples_samplingrate)
    # save the index
    index.append(mfccs.shape[0])

    for i in range(len(data)):
        if i == 0:
            continue
        piece = data[i]
        samples = piece['samples']
        samples_samplingrate = piece['samplingrate']
        # mfcc
        mfcc = proto.mfcc(samples,samplingrate=samples_samplingrate)
        mfccs = np.vstack((mfccs, mfcc))
        # save the index
        index.append(mfccs.shape[0])


    g = GaussianMixture(n_components=4)
    # Generate random observations w
    g.fit(mfccs)

    # calculate the posterior
    # resp = g.predict_proba(mfccs)
    # # get the largest probability and change array to list
    # index_resp = resp.argmax(axis=1).tolist()

    # use the predict function
    resp = g.predict(mfccs)
    index_resp = resp.tolist()

    cluster_mfccs = []
    # save the same samples
    for i in range(len(index)):
        if i == 0:
            cluster_mfccs.append(index_resp[0:index[0]])
            continue
        cluster_mfccs.append(index_resp[index[i-1]:index[i]])

    # get the most times
    result = []
    for i in range(len(cluster_mfccs)):
        c = Get_Most(cluster_mfccs[i])
        result.append(c)

    # show the result
    x = range(0,len(data))
    plt.scatter(x, result)
    plt.show()


