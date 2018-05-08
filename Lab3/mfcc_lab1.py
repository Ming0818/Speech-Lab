# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------

#************************* ****
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *
from scipy.fftpack import *
from tools import *
from scipy.fftpack.realtransforms import *
#*************************

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    mspec = logMelSpectrum(spec, samplingrate)
    ceps = cepstrum(mspec, nceps)
    return lifter(ceps, liftercoeff)


def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples. 20ms
        winshift: shift of consecutive windows in samples 10ms
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    samples_len = len(samples)
    temp = np.zeros(winlen)
    N = 0
    if samples_len > winlen:
        temp = samples[0:winlen]
        N=N+1
        while(True):
            # compute start and end indexes
            end = winlen + N*(winlen-winshift)
            start = end-winlen
            # if it exceeds the total length, drop the last part
            if(end > samples_len):
                break
            # if it's not the last part, append the window to the array
            temp = np.vstack((temp, samples[start:end]))
            N=N+1
    else:
        temp = np.hstack((samples, np.zeros(winlen - samples_len)))
    return temp

def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    preemph = np.zeros(input.shape[1])
    # a = [1,0,0,...,0]
    a = np.zeros(input.shape[1])
    a[0]=1
    # b = [1,-p,0,0,...,0]
    b = np.zeros(input.shape[1])
    b[0]=1
    b[1] = -p
    # y[n] = x[n]-ax[n-1]
    # lfilter: a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M] - a[1]*y[n-1] - ... - a[N]*y[n-N]
    # so, 1*y[n] = 1*x[n] + (-p)*x[n-1]
    for i in range(input.shape[0]):
        # a = 0 IIR -> FIR
        sig = lfilter(b, a, input[i],axis=-1, zi=None)
        preemph = np.vstack((preemph,sig))
    return preemph[1:]

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    window = hamming(input.shape[1],sym=False)

    # plot the window shape
    # plt.plot(window)
    # plt.show()

    windowed = np.zeros(input.shape[1])
    for i in range(input.shape[0]):
        windowed = np.vstack((windowed, input[i] * window))
    return windowed[1:]

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    spec = np.zeros(nfft)
    for i in range(input.shape[0]):
        f_t = fft(input[i], nfft , axis=-1, overwrite_x=False)
        f_t = f_t.real**2+f_t.imag**2
        spec = np.vstack((spec,f_t))
    return spec[1:]

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input.shape[1]
    mspec = []
    fbank = trfbank(samplingrate, nfft, lowfreq=133.33, linsc=200 / 3., logsc=1.0711703, nlinfilt=13, nlogfilt=27,
                    equalareas=False)
    # plt.plot(fbank.T)
    # plt.show()
    for i in range(input.shape[0]):
        result = np.dot(input[i].reshape(1,-1),fbank.T)
        f_result = np.log(result)
        if(i == 0):
            mspec = f_result
        else:
            mspec = np.vstack((mspec, f_result))
    return mspec

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    ceps = dct(input, type=2, axis=1, norm='ortho')[:, 0: nceps]
    return ceps


def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """
    # compute local distances
    locD = np.zeros((x.shape[0],y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            locD[i][j] = dist(x[i],y[j])
    # print(locD)

    # dynamic programming
    accD = np.zeros((x.shape[0], y.shape[0]))
    for h in range(accD.shape[0]):
        for k in range(accD.shape[1]):
            if h == 0 and k == 0:
                accD[h][k] = locD[h][k]
                continue
            if h == 0:
                accD[h][k] = locD[h][k] + accD[h][k - 1]
                continue
            if k == 0:
                accD[h][k] = locD[h][k] + accD[h - 1][k]
                continue
            accD[h][k] = locD[h][k] + min(accD[h - 1][k], accD[h - 1][k - 1], accD[h][k - 1])

    return accD[accD.shape[0]-1][accD.shape[1]-1]
