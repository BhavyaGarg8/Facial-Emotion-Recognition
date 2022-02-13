# Facial-Emotion-Recognition

## Abstract:
As the time is passing, science is approaching more and more technologies. 
Some people are aiming to develop some machines that can interact with 
human-beings. Any model that will help the machine to classify the emotion of 
a human face will be immensely helpful for it. Here we are doing something 
similar, our model will detect **facial expression** using **CNN**.

## Introduction:
All humans interact with each other and shares their emotions with 
each other. Most importantly they use their **Facial Expressions** to 
share their inner emotions.
In this project we are creating a model using **Deep Learning –
Convolution neural network**, that can detect the face expression, 
more specifically it classifies it into one of the emotions: 0 [anger], 
1[disgust], 2[fear], 3[Happy], 4[Sadness], 5[surprise], 6[Neutral].

## Data-Set:
We will use a standard data set **FER-2013** available on the Kaggle 
Website.
This dataset is grayscale with dimension of images as 48*48 with a 
total of 36000 samples. Each image is classified into one of the seven 
emotions indexed from 0 to 6

## Method and Techniques:

We have built a CNN model in python using deep learning.
We have used **Kera’s** imported from **TensorFlow module**.
Our CNN model basically contains **4 layers**;

### 1.Input Layer:
The csv file containing pixel values of each image in given as a input. 
As the **pixel** values are given in string form, we need to convert them 
into float values and dimension the pixel values in 2D array with 
**normalized values**, using suitable python code.

### 2.Convolution Layers:
The main work is done by these layers. In short if we see, then a 
facial image will have many features like Eyes, Nose, lips, eyebrows, 
forehead etc. Every feature will correspond to a specific type for a 
corresponding facial expression.
The CNN will map different features with the help of neural network
(mathematical expression).

* First Block will contain **64** different filters and a **“ReLU”** activation, 
followed by a **Max pool** (stride=2) and a **normalization** layer.

* Second Block contains another convolution layer with **128** filters and 
“ReLU” activation, followed by a max pooling and a normalization 
layer.

* Third Block contains another convolution layer with 128 filters and 
“ReLU” activation, followed by a max pooling and a normalization 
layer.
* Fourth Block contains another convolution layer with 528 filters and 
“ReLU” activation, followed by a max pooling and a normalization 
layer.
Adding Normalization layer gives a better accuracy.
Finally modelled was **flattened** into a **1D** array.

### 3.Dense Layer:
The flattened layer will now have size (4096).
Thus, two more **dense layers** were added with 4096(“ReLU” 
activation) and 7(“SoftMax” activation) neurons respectively. As 
there are only 7 possible emotions therefore last layer have only 7 
neurons. And a **dropout of 0.3** was added between these layers for 
better results.

### 4. Output Layer:
This layer outputs the **probability** distribution of all emotions for a 
particular image data. The highest probability emotion is considered 
as predicted expression.

### Result and Analysis:
Before training, dataset was divided into training and testing sets.
20% data was transferred for** testing** and rest will be used for **training** 
process.
Now after almost 20 **epochs** our model will give almost 60% 
accuracy.
Now we plotted **confusion matrix**.

After analyzing we see that certain expressions are predicted with 
high accuracy.
Say it is easy for humans to find if a person’s expression is happy or 
not. But for other expressions, sometimes it is not possible for us to 
guess the right emotion.
Thus, Happy and surprise expressions had a great accuracy over 
others.
Our model sometime confused between fear or neutral when the 
actual was sad and vice versa for a large number of sets.
Thus, it had a low accuracy in these expressions.

### Conclusion:
* Our model achieved almost 60% accuracy. 
* With this project we learned some technical things like how to suitably increase filters in 
layers, when to do max pooling and normalization to achieve a better
accuracy.
*Dividing the dataset is also necessary so that our model does not get biased to the given input set.
