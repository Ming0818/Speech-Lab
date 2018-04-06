import proto
import matplotlib.pyplot as plt
from tools import *
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

def getData():
    data = np.load('lab1_data.npz')['data']
    globalD = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            mfccs_i = proto.mfcc(data[i]['samples'])
            mfccs_j = proto.mfcc(data[j]['samples'])
            globalD[i][j] = proto.dtw(mfccs_i, mfccs_j)
            # print(globalD[i][j])
    np.savetxt('distance', globalD)

if __name__== "__main__":
    globalD = np.loadtxt('distance')

    # display matrix with pcolormesh
    # plt.pcolormesh(globalD)
    # plt.show()

    # hierarchical clustering
    Z = linkage(globalD, 'complete')
    # get label
    data = np.load('lab1_data.npz')['data']
    labels = tidigit2labels(data)
    # plot with labels
    dn = dendrogram(Z,labels=labels)
    plt.show()