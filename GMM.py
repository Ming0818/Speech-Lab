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
    mfcc_orignal = []
    index = []

    piece = data[0]
    samples = piece['samples']
    samples_samplingrate = piece['samplingrate']
    # mfcc
    mfccs = proto.mfcc(samples, samplingrate=samples_samplingrate)
    # save the index
    index.append(mfccs.shape[0])
    # orignal mfcc
    mfcc_orignal.append(mfccs)

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
        # original mfcc
        mfcc_orignal.append(mfcc)

    g = GaussianMixture(n_components=32)
    # Generate random observations w
    g.fit(mfccs)

    # calculate the posterior(whole matrix)
    resp = g.predict_proba(mfccs).tolist()

    # calculate the posterior(one by one)
    posterior_result = []
    for i in range(len(index)):
        r = g.predict_proba(mfcc_orignal[i]).tolist()
        posterior_result.append(r)

    cluster_mfccs_result = []
    for i in range(len(index)):
        if i == 0:
            cluster_mfccs_result.append(resp[0:index[0]])
            continue
        cluster_mfccs_result.append(resp[index[i-1]:index[i]])

    # compare
    # for i in range(len(index)):
        # print( (abs(np.array(cluster_mfccs_result[i]) - np.array(posterior_result[i]))<0.0001 ).all())

    for i in range(len(index)):
        plt.subplot(2,22,i+1)
        plt.title("{}".format(i))
        # plt.imshow(cluster_mfccs_result[i])
        plt.imshow(posterior_result[i])
        plt.axis('off')
    plt.show()
