import proto
import matplotlib.pyplot as plt
from tools import *

if __name__== "__main__":
    data = np.load('lab1_data.npz')['data']
    piece = data[0]
    samples = piece['samples']
    samples_samplingrate = piece['samplingrate']
    # mfcc
    mfccs = proto.mfcc(samples, samplingrate=samples_samplingrate)
    # mspec
    mspecs = proto.mspec(samples, samplingrate=samples_samplingrate)
    for i in range(len(data)):
        if i == 0:
            continue
        piece = data[i]
        samples = piece['samples']
        samples_samplingrate = piece['samplingrate']
        # mfcc
        mfcc = proto.mfcc(samples,samplingrate=samples_samplingrate)
        mfccs = np.vstack((mfccs, mfcc))
        # mspec
        mspec = proto.mspec(samples, samplingrate=samples_samplingrate)
        mspecs = np.vstack((mspecs, mspec))
        # plot
        # plt.subplot(11, 4, i+1)
        # plt.pcolormesh(mfcc.T)
        # plt.axis('off')
    print(mfccs.shape)
    # plt.pcolormesh(mfccs.T)
    # plt.pcolormesh(mspecs.T)
    # plt.figure()
    corrcoef = np.corrcoef(mfccs.T)
    corrcoef_before = np.corrcoef(mspecs.T)
    print(corrcoef)
    print(corrcoef_before)
    plt.subplot(1, 2, 1)
    plt.pcolormesh(corrcoef)
    plt.subplot(1, 2, 2)
    plt.pcolormesh(corrcoef_before)
    plt.show()