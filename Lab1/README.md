# DT2119 Lab 1: Feature extraction

Ci Li cil@kth.se

Mengdi Xue mengdix@kth.se

## Overall process

The overall process is as the following figure:

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/process.png" width=300/>

The signal:

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/signal.png" width=600/>

## 4.1 enframe

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/enframe2.png" width=600/>

Data.samplingrate= 20000 Hz. 

The window length of 20 ms, the number of signal in each window is 20kHz * 20 ms = 400

The shift is 10 ms, the number of overlap signal in one window is 20kHz * 10 ms = 200

## 4.2 Pre-emphasis

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/preemp2.png" width=600/>

The lfilter function in scipy.signal:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20a%5B0%5D%2Ay%5Bn%5D%20%3D%20b%5B0%5D%2Ax%5Bn%5D%20%2B%20b%5B1%5D%2Ax%5Bn-1%5D%20%2B%20...%20%2B%20b%5BM%5D%2Ax%5Bn-M%5D%20-%20a%5B1%5D%2Ay%5Bn-1%5D%20-%20...%20-%20a%5BN%5D%2Ay%5Bn-N%5D" style="border:none;">

And in our experiment:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20y%5Bn%5D%20%3D%20x%5Bn%5D-%5Calpha%20x%5Bn-1%5D" style="border:none;">


So the filter coefficient is as following:

a is an array with the length of input signal, and a[0] is 1, others are 0, i.e. ```a = [1, 0, 0, ..., 0]```

b is an array with the length of input signal, and b[0] is 1, b[1] is -0.97, others are 0, i.e. ```b = [1, -0.97, 0, 0, ..., 0]```

Pre-emphasis is similar as a high frequency filter in order to highlight the high frequency part which is suppressed by the pronunciation system.

## 4.3 windowing

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/window.png" width=400/>

The reason why this windowing should be applied to the frames of speech signal:

1)used as one of many windowing functions for smoothing values. 

2)To reduce spectral leakage compared with other windowing function.

after windowing:

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/afterwindow.png" width=600/>

## 4.4 FFT

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/FFT2.png" width=600/>

According to Sampling Theorem:
F_max = Samplingrate /2 (10kHz) in order to avoid spectral aliasing

## 4.5 Mel filterbank log spectrum

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/trfbank.png" width=600/>

The distribution of the filters along the frequency axis:
There are totally 40 filters. From the distribution we can see that there are more triangle filters in low frequency and less triangle filters in high frequency. Also the amplitude of the triangles is decreasing from low frequency to high frequency while the width of the triangles is increasing. It can remain more low frequency signals and drop high frequency which meets the non-linear human ear perception of sound, by being more discriminative at lower frequencies and less discriminative at higher frequencies. 

The resulting filterbank outputs

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/Mel%20filterbank%20log%20spectrum2.png" width=600/>

## 4.6 Cosine Transform and Liftering

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/mfcc2.png" width=600/>

**Cosine Transform**

After DCT, the energy will focus on the media and low frequency part. So we can use DCT to directly get the low frequency part from the spectrum.

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/lmfcc2.png" width=600/>

**Liftering**

Liftering Function is used to correct the range of the coefficients.

## 4.7 Compare the liftered MFCCs of different utterances
Four utterances with the same digit "7"
![](https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/compare_mfcc_different_utter.png)

Four utterances with two digit "7" and "8"
![](https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/compare_mfcc_different_utter_78.png)


## 5 Feature Correlation

![](https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/compare_cepstrum.png)

The left image is the correlation coefficients after MFCC features. The right image is the correlation coefficients after Mel filterbank features.

**Q:** Are features correlated?

**A:** No. We can see from the left image that features are only correlated with themselves.

**Q:** Is the assumption of diagonal covariance matrices for Gaussian modelling justified? Compare the results you obtain for the MFCC features with those obtained with the Mel filterbank features ('mspec' features).

**A:** Yes. The covariance matrices is diagonal. The diagonal elements of the matrix is 1.0 while others are small numbers around zero. However, the results from Mel filterbank features are not diagonal. It is because that we applied Discrete Cosine Transform (DCT) to decorrelate the filter bank coefficients and obtained filter banks with 13 coefficients.

## 6 Comparing Utterances
The result of the global distance between the two sequences:

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/compare_utterances.png" width=600/>

**Q:** Does the distance separate digits well even between different speakers?

**A:** Yes. We can see from the figure that only the two adjacent utterances and the utterances themselves are highly correlated (in very dark color). The same digits between different speakers are less similar than between the same speakers. We can see two diagonals with offset in the figure that represent same digits between different speakers.

The result of the hierarchical clustering on the distance matrix:

![](https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/clustering_labels.png)

From the result, we can see that most of the utterances that represent the same digits from the same speaker are clustered together. Also, there are two digits that are successfully clustered together: 2 and 3.

## 7 Explore Speech Segments with Clustering

The result for digit 3 with 4, 8, 16 and 32 components are as following:

![](https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/same_digit_all_posterior.png)

The result for digit 2 and 3 with 4, 8, 16 and 32 components are as following:

![](https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/diff_digit_all_posterior.png)

The result for digit 2, 3, 4, 5 with 4 components are as following:

<img src="https://github.com/Celiali/Speech-lab1-MFCC/blob/master/Lab1/figure/diff_digit_4posterior_8-15.png" width=600/>

**Q:** Are those classes a stable representation of the word if you compare utterances from different speakers? Do you see a relationship between the most active component and the kind of sound at any particular time? Do utterances containing the same spoken word by different speakers contain the same or similar evolution of posteriors?

**A:**
We can see from the first figure that when components increase, the stability of the classes decreases. With 4 components, the posteriors are as the following sequences: blue and orange for about 0-20, red for 25-50, orange for 50-65, and blue for 65-80. The trends are the same while there are offsets for each periods. So we can conclude that they have similar evolution of posteriors. But for different digits, the length and sequence of the posteriors are much more different.

But when the component number increases to 16 and 32, the posteriors that form an utterance is more various between different speakers.
