import proto
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *
from scipy.fftpack import *
from tools import *
from scipy.fftpack.realtransforms import *

if __name__== "__main__":
    data = np.load('lab1_data.npz')['data']
    example = np.load('lab1_example.npz')['example'].item()
    samples = example['samples']
    samples_samplingrate =example['samplingrate']

    # plot the original signal
    plt.subplot(8, 1, 1)
    plt.plot(samples)
    plt.axis('off')
    # plt.show()

    # ------------enframe-------------------
    winlen=400
    winshift= 200
    frames=proto.enframe(samples, winlen, winshift)

    # varify with example['frames']
    # compare_f = example['frames']
    # result_f = sum(sum(frames-compare_f))
    # print(result_f)

    # plot the array
    plt.subplot(8, 1, 2)
    plt.pcolormesh(frames.T)
    plt.axis('off')
    # plt.show()

    #-------------preemp---------------------
    preempcoeff = 0.97
    preemph = proto.preemp(frames, preempcoeff)

    # varify with example['preemph']
    # compare_p = example['preemph']
    # print((preemph == compare_p).all())

    # plot the array
    plt.subplot(8, 1, 3)
    plt.pcolormesh(preemph.T)
    plt.axis('off')
    # plt.show()

    # -------------window---------------------
    windowed = proto.windowing(preemph)

    # plot the array
    plt.subplot(8, 1, 4)
    plt.pcolormesh(windowed.T)
    plt.axis('off')
    # plt.show()

    # varify with example['windowed']
    # compare_w = example['windowed']
    # print((windowed == compare_w).all())

    #---------------FFT----------------------
    nfft = 512
    spec = proto.powerSpectrum(windowed, nfft)

    # varify with example['windowed']
    # compare_s = example['spec']
    # print((abs(spec - compare_s)<0.0000001).all())

    # plot the array
    plt.subplot(8, 1, 5)
    plt.pcolormesh(spec.T)
    plt.axis('off')
    # plt.show()

    #---------------Mel filterbank log spectrum------
    mspec = proto.logMelSpectrum(spec, samples_samplingrate)

    # varify with example['mspec']
    # compare_m = example['mspec']
    # print((abs(mspec - compare_m) < 0.0000001).all())

    # plot the array
    plt.subplot(8, 1, 6)
    plt.pcolormesh(mspec.T)
    plt.axis('off')
    # plt.show()

    # --------------Cosine Transofrm------------------
    coeff =  np.arange(13).reshape(1,-1)
    nceps = coeff.shape[1]

    ceps = proto.cepstrum(mspec, nceps)

    plt.subplot(8, 1, 7)
    plt.pcolormesh(ceps.T)
    plt.axis('off')
    # plt.show()

    # varify with example['mfcc']
    # compare_c = example['mfcc']
    # print((abs(ceps - compare_c) < 0.0000001).all())

    #------------------lmfcc--------------------------
    lmfcc = lifter(ceps, lifter=22)

    plt.subplot(8, 1, 8)
    plt.pcolormesh(lmfcc.T)
    plt.axis('off')
    # plt.show()

    # varify with example['lmfcc']
    # compare_l = example['lmfcc']
    # print((abs(lmfcc - compare_l) < 0.0000001).all())

    plt.show()