# Lab3: Phoneme Recognition with Deep Neural Networks

Ci Li cil@kth.se

Mengdi Xue mengdix@kth.se

## 4 Preparing the Data for DNN Training

### 4.1 Target Class Definition

We create a list of unique states for reference, to make sure that the output of the DNNs always refer to the right HMM state. The stateList looks like this:
```
['ah_0', 'ah_1', 'ah_2', 'ao_0', 'ao_1', 'ao_2', 'ay_0', 'ay_1', 'ay_2', ...,
 ..., 'w_0', 'w_1', 'w_2', 'z_0', 'z_1', 'z_2']
 ```

### 4.2 Forced Alignment

In this step, we need to know the right target class for every time step to train deep neural networks. So we build a combined HMM concatenating the models for all the phones in the utterance, and then run the Viterbi decoder  to recover the best path.

**We use file "z43a.wav" to test our code. We visualise the speech file "z43a" and the transcription.**

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/wavesurfer.png)

**Q:** Is the alignment correct? What can you say observing the alignment between the sound file and the classes?

**A:** From the result, we can see that the alignment is correct. For example, it begins with sil_0, transfers to sil_1 and then slowly transfers to z_0, z_1, etc.

### 4.3 Feature Extraction

In this step we extract features and targets for the whole database and saved them in .npz files.

### 4.4 Training and Validation Sets

We split the training data into a training set (roughly 90%) and validation set (remaining 10%). Here is **how we selected the two datasets:**

The whole train set contains 8623 datas, the number of man samples is 4235 and the number of woman samples is 4388. And each person has about 77 utters. If we want to split the training data into a training set (roughly 90%) and validation set (remaining 10%), the number of validation set is about 862. Moreover, each speaker is only included in one of the two sets. So the number of speakers in validation set is about 12. And we choose the first **6 men** speakers and the first **6 women** speakers to combine the validation set.

### 4.5 Dynamic Features

For dynamic features, we need to stack 7 features symmetrically distributed around the current time step.

In order to have the dynamic features, we extract and reverse ```feature[:,1:3]```, ```feature[:,-4:-2]``` and stack them at the beginning and end of the feature, like this:```hstack(reverse(feature[:,1:3]),feature,reverse(feature[:,-4:-2]))```

### 4.6 Feature Standardisation

In this step, we choose to **normalise over the whole training set**. We concate all the lmfcc, mspec and targets respectively for training set, validation set and test set respectively each for _non dynamic features_ and _dynamic features_.

Then we use ```StandardScaler``` from ```sklearn.preprocessing``` to normalise over the whole training set, save the normalisation coefficients and reuse them to normalise the validation and test set.


**Q:** What is the the implications of the three different strategies to normalize? What will happen with the very short utterances in the isolated digits files if normalising each utterance individually?

**A:** The environment when we collect each utterances may differ. So normalising each utterance individually can't help to standardise the data. It is the same if we normalize over each speaker separately. Only normalizing over the whole training set can help us to reduce the effect of different environments and speakers.

Also, if normalising each utterance, we can't decide which mean and variance to use for normalising validation set and test set.

If the utterance is very short, then the normalized results will differ more between utterances.

## 5 Phoneme Recognition with Deep Neural Networks

First before entering the DNN, we convert feature arrays to 32 bits floating point format because of the hardware limitation in most GPUs and the target arrays into the Keras categorical format.

Then we use Keras to build the DNN for training. We build a ```Sequential``` model and use ```Dense``` to add layer to the model. The **loss function** we use is ```categorical_crossentropy```. The **optimizer** is ```adam```(learning rate is 0.002).

#### Settings of DNN

**Q:** Define the proper size for the input and output layers depending on your feature vectors and number of states.

**A:** The input layer is depending on the input size of feature vectors. If the input includes dynamic feature, the input layer size of lmfcc and mspec is 91 and 280. Otherwise, the input layer size of lmfcc and mspec is 13 and 40.

The output layer is 61(the number of states).

To hidden layer: The first layer is 256. The second layer is 128. The third layer is 64. The fourth layer is 32.

**Q:** Choose the appropriate activation function for the output layer, given that you want to perform classification.

**A:** The activation function for output layer is **Softmax**. The activation function for hidden layer is **ReLu**.

**Q:** Why you chose the specific activation and what alternatives there are?

**A:** The activation function for hidden layer is ReLu instead of Sigmoid. Because Sigmoid uses power operation which is slower than ReLu. Another reason is that Sigmoid can cause gradient vanishing while ReLu will not.

#### Fitting model

Then we fit the model to perform training. The batch size we used is 256. And we use 10 epoches.

**Q:** What is the purpose of the validation data?

**A:** The validation data set is used to avoid overfitting.

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/epoch_training_val.png)

**Q:** What can you say comparing the results on the training and validation data?

**A:** We can see that their loss are both decreasing, so the model didn't overfit, since a model is overfitting after the validation loss starts to increase. Also, the accuracy of them are all increasing. But different with training, validation loss and accuracy has fluctuations because they are new data.


### 5.1 Detailed Evaluation

We use ```predict``` and ```evaluate``` to evaluate the output of the network.

***First***, we plot the posteriors and the target values for an example utterance(oo7oa.wav) as following. (Here we use the network contains three hidden layers and trained with the data without dynamic features.)

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/posterior_oo7oa.png)

**Q:** What properties can you observe?

**A:** From the result, we can see that the general trend is the same between the prediction and targets. What's more, you can see the distributions of these four digits are very clear.

***Second***, we evaluate the classification performance in four ways.

![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/evaluation.png)

For the first two types of evaluations, we also compute the confusion matrices. The figures below is the example under the 3-hidden-layer network trained with data without dynamic features.

The confusion matrix of lmfcc
![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/confusion_matrix_lmfcc_3layers.png)

The confusion matrix of mspec
![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/confusion_matrix_mspec_3layers.png)

From the result, we can see that the color in the diagnol is blighter which means most of the phoneme or states are correctly recognized. The blightest part focus on 39th,40th,41th state and 13th phoneme which represent the state related to silence.


### 5.2 Possible questions

1. What is the influence of feature kind and size of input context window?

    When we use dynamic features with the same network settings, it is not better than without dynamic features. We guessed if doubling the size of hidden layer can help, but we get ```58.58%``` with the hidden layer size of ```256, 128, 64```	and ```57.62%``` with hidden layer size of ```512, 256, 128```. So it doesn't help.

2. What is the purpose of normalising (standardising) the input feature vectors depending on the activation functions in the network?

    Normalising is to avoid the influnces of the environments and other aspects for DNN to train better.

3. What is the influence of the number of units per layer and the number of layers?

    More units per layer and more number of layers can increase the result accuracy. But too many units or too many layers may cause the network to overfit. Under our parameters, the result of 3 layers are the best.

4. What is the influence of the activation function (when you try other activation functions than ReLU, you do not need to reach convergence in case you do not have enough time)

    The result of Sigmoid function and ReLU function are similar. Their loss figure are as following:
    
    ![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/compare_sigmoid_relu.png)
    
    The test accuracy of Sigmoid (3 layers, lmfcc, eta=0.002)is ```60.14%``` and ReLU is ```59.76%```. But ReLU is much faster because it doesn't need to compute power operations.

5. What is the influence of the learning rate/learning rate strategy?

    When we increase the learning rate, the result accuracy increases. But too high learning rate may cause overfitting so if we want to have a higher learning rate we may need some regularization. The result of different learning rate on the whole datasets is as following. Their test accuracy are: 16.83%, 59.76%, and 61.01%. We can see that 0.0002 is more suitable for our data.
    
    ![](https://github.com/Celiali/Speech-Lab/blob/master/Lab3/figure/compare_learningrate.png)

6. How stable are the posteriograms from the network in time?

    The result of posteriors during continuous time does not change much, or slowly change from one phoneme to another.

7. How do the errors distribute depending on phonetic class?

    We can see from the confusion matrix that there are more errors with similar phonemes. For example, there are more errors of recognizing 'ah' as 'ay'.
