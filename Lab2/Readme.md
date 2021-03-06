# DT2119 Lab 2: HMM with Gaussian Emissions

Ci Li cil@kth.se

Mengdi Xue mengdix@kth.se

## Step-by-step probability calculations for utterance 'o' in the example

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/stepbystep_calculations.png)

## 4.1 Gaussian emission probabilities

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/loglikihood.png)
We can see that the starting likelihood and the finishing likelihood is the same distribution since they are both corresponding to 'sil'. Between two phonemes, the distribution has a gradual transition and corresponding to 'ow'.
Also we can see that the distributions between two different digits are different.

## 4.2 Forward algorithm

The meaning of alpha is the probability of the seeing sequence y{1}, y{2},...,y{t} being in state i at time t, as following:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/d7ad4d959c520ccc1d473c4cd70626a1509f731f)

We calculated the forward probablity according to the following formulas:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20log%5Calpha_%7B0%7D%28j%29%3Dlog%5Cpi_j%2Blog%5Cphi_j%28x_0%29" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20log%5Calpha_%7Bn%7D%28j%29%3Dlog%28%5Csum_%7Bi%7Dexp%28log%5Calpha_%7Bn-1%7D%28i%29%2Blog%20a_%7Bij%7D%29%29%2Blog%5Cphi_j%28x_n%29" style="border:none;">

Then, we derived the likelihood ```P(X|θ)``` of the whole sequence ```X = {x0, x1, ..., xN-1}``` as following:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20P%28X%7C%5Ctheta%29%3D%5Csum_%7Bi%3D1%7D%5E%7BM%7D%5Calpha_N%28i%29" style="border:none;">

The formula in log domain is as following:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20logP%28X%7C%5Ctheta%29%3Dlog%28%5Csum_%7Bi%3D1%7D%5E%7BM%7Dexp%28log%5Calpha_N%28i%29%29%29" style="border:none;">

We applied it on the example data and got the likelihood of ```-3769.415```, which is the same as ```example['loglik']```.

here is the result of example
![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/forward.png)


At last, we applied our forward algorithm on all the 44 utterances with each of the 11 HMM models 

<img src="https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/score_utter.png" width=600/>

And then we took the maximum likelihood model as winner, the result is as following:

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/forward_44with11.png)

We can see that all the 44 results are right.

## 4.3 Viterbi algorithm

Viterbi algorithm is a dynamic programming algorithm to find the most likely sequence of hidden states(Viterbi path) that results in a sequence of observed events. It is to backtrace to find the state sequence of the most probable path as following:

![](https://img-blog.csdn.net/20140530160136578?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvYWJjamVubmlmZXI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

We calculated the viterbi probability according to the following formulas:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20logV_0%28j%29%3Dlog%5Cpi_j%2Blog%5Cphi_j%28x_0%29" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20logV_n%28j%29%3D%5Cmax_%7Bi%3D0%7D%5E%7BM-1%7D%28logV_%7Bn-1%7D%28i%29%2Bloga_%7Bij%7D%29%2Blog%5Cphi_j%28x_n%29" style="border:none;">

And then find the path according to the following formulas:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20B_n%28j%29%3Darg%5Cmax_%7Bi%3D0%7D%5E%7BM-1%7D%28logV_%7Bn-1%7D%28i%29%2Bloga_%7Bij%7D%29" style="border:none;">

We can see that the path starts from state 0 and ends with state 7 since state 0,1,2,6,7,8 are corresponding to 'sil'. And from frames 11 to frames 50, the path starts from state 3 to state 5 corresponding to 'ow'. We can see the path rise slowly because each state tends to stay at the same state more. While in the middle part, the path stays at the same state for a while and then transits to the next state.

here is the result of example
![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/viterbi.png)

At last, we applied our viterbi algorithm on all the 44 utterances with each of the 11 HMM models, the result is as following:

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/viterbi_44with11.png)

We can see that there are no mistakes. It is the same with the result we got in the previous section.

## 4.4 Backward algorithm

The meaning of beta is the probability of the ending partial sequence y{t+1},...,y{T} given starting state i at time t, as following:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/ffb576763df197cd5a712f7624970f5193feddb9)

We calculated the backward probability according to the following formulas:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20log%5Cbeta_%7BN-1%7D%28i%29%3D0" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20log%5Cbeta_%7Bn%7D%28i%29%3Dlog%28%5Csum_%7Bj%7Dexp%28log%20a_%7Bij%7D%2Blog%5Cphi_j%28x_%7Bn%2B1%7D%29%2Blog%5Cbeta_%7Bn%2B1%7D%28j%29%29%29" style="border:none;">

Then, we derived the likelihood ```P(X|θ)``` of the whole sequence ```X = {x0, x1, ..., xN-1}``` as following:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cbeta_0%28i%29%3DP%28x_1%2C%20...%2C%20x_%7BN-1%7D%7Cz_0%3Ds_i%2C%20%5Ctheta%29" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cpi_i%20%3D%20P%28z_0%3Ds_i%29" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cphi_i%28x_0%29%3DP%28x_0%7Cz_0%3Ds_i%29" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20P%28x_0%2C%20x_1%2C%20...%2C%20x_%7BN-1%7D%2C%20z_0%7C%5Ctheta%29%20%3D%20P%28x_1%2C%20...%2C%20x_%7BN-1%7D%7Cz_0%3Ds_i%2C%20%5Ctheta%29P%28x_0%7Cz_0%3Ds_i%29P%28z_0%3Ds_i%29%3D%5Cbeta_0%28i%29%5Cpi_i%5Cphi_i%28x_0%29" style="border:none;">

The formula in log domain is as following:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20P%28X%7C%5Ctheta%29%20%3D%20log%28%5Csum_%7Bi%3D1%7D%5E%7BM%7Dexp%28log%5Cbeta_0%28i%29%2Blog%5Cpi_i%2Blog%5Cphi_i%28x_0%29%29%29" style="border:none;">

Then we calculated the likelihood ```P(X|θ)``` according to the above formula and it is also ```-3769.415```, which is the same as ```example['loglik']```.

here is the result of example
![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/backward.png)

## 5.1 State posterior probabilities

The meaning of gamma is the probability of being in state i at time t given the observed sequence Y and the parameters theta, as following:

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/5ecf98a64ec3cdf949076c545223d982bff56748)

We calculated the state posterior probabilities according to the follow formulas:

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/state1.png)

here is the result of example
![](https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/state_posterior.png)

When we sum the posteriors (in linear domain) for each state along the time axis, the results are as following:

```
3.73001152   3.0047182
4.45622881   5.58497619
13.5585008   15.80238479
12.763586    12.02229947
0.07729421
```

The meaning of the above results is the sum of the probability for each state for all observations, which is the denominator in the formula used in 5.2(sum of gamma).

And the sum of each time step is 1. When we sum over both states and time steps, the result is: ```70.9999999998```, which is nearly 71 and the same length of the observation sequence. Because for each time step, the sum of the posteriors are 1, so for 71 time steps, the overall sum is 71.

## 5.2 Retraining the emission probability distributions

At last, since we concatenate the HMM models, we need to do the retraining to better fit the models with Baum–Welch algorithm, given a sequence of feature vectors and the array of state posteriors probabilities.

We calculated the probablities according to the following formulas:

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5Cmu_j%3D%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cgamma_n%28j%29x_n%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cgamma_n%28j%29%7D" style="border:none;">

<img src="http://chart.googleapis.com/chart?cht=tx&chl=%20%5CSigma_j%3D%5Cfrac%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cgamma_n%28j%29x_n%20x_n%5ET%7D%7B%5Csum_%7Bn%3D1%7D%5E%7BN%7D%5Cgamma_n%28j%29%7D%20-%20%5Cmu_j%5Cmu_j%5ET" style="border:none;">

utterance ```data[10]``` starting with ```wordHMMs['4']```:

<img src="https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/4-4.png" width=180/>

utterance ```data[10]``` starting with ```wordHMMs['9']```:

<img src="https://github.com/Celiali/Speech-Lab/blob/master/Lab2/figure/4-9.png" width=180/>

We can see that they both take 6 rounds to converge. But the first model starts with higher likelihood and converges to a higher likelihood because it is the right model.
