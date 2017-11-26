## Follow me project writeup

### Environment:

For this project I decided to build my own GPU platform to experiment with. Besides my main graphic card, I got extra NVIDIA GTX980. Setting up CUDA on Ubuntu 14.04 deserves a separate writeup:), but once i got it working, I gained more than x10 boost in calculation speed.

[gtx]:./images/GTX.jpg
![gpu][gtx]


### Network architectures

[2lnn]:./images/2_layers_nn.jpg
[3lnn]:./images/3_layers_nn.jpg

##### 2-Layer deep architecture

In this project I experimented with 2 neural network architectures: 2-layer (2 encoders + 2 decoders) and 3-layer (3 encoders + 3 decoders);
Layer sizing is shown on the pictures below

![nn 1][2lnn]


##### 3-Layer deep architecture

![nn 2][3lnn]


**For some reasons 2-layer FCN produced much better results, on some runs score went up to 44%, whereas 3-layer network gave at most 39%. I suppose I just did not do enough tuning, and dataset was not large enough.**
**Submitted score is: 0.417897433983**

### Project components

1. encoder_block(input_layer, filters, strides):
Function builds a convolution layer. This layer is learning learn features with spatial
information by sliding at a window with a certain kernel size.
**[Rev1]** Encoder convolution layers extract features from layer to layer -> from basic to more complex object, for example, the first layer might only look at lines or edges, next layer might look for shapes, and the following layers look for more complex objects (faces, street signs etc).

  * a. input_layer - input from previous layer 
  * b. filters – amount of features that will be extracted 
  * c. strides – number of strides step we slide the window.

2. decoder_block(small_ip_layer, large_ip_layer, filters):
This is essentially a reverse of the encoder layer. The purpose of this layer is to
upscale the image back to its original size and extract the encoded information the net found
using the encoder blocks. 
**[Rev1]** However, when we decode the output of the image back to the original image size some data may be lost. Skip connection helps to retain this information. Each layer of the Decoder contains a skip connection to the corresponding encoder layer. Skip connection connects the output of one layer to the input of the other. As a result, the network is able to make more precise segmentation decision.

  * a. small_ip_layer – input from the previous layer
  * b. large_ip_layer – the skip layer
  * c. filters – amount of features that are extracted


3. 1x1 conv layer. 

**[Rev1]** Difference between 1x1 convolution and fully connected layer
This layer extracts non linear features for each pixel in a
layer, and acts basically like FCN (fully connected network) for each pixel. Like in the encoder layers, the weight sharing gives allot of pixels for our network to train on no matter where on the image they reside. In other words difference is that the 1x1 convolution layer preserves spatial
information as opposed to a fully connected layer. Since spatial information is lost it makes network not usable for pixel-wise classification

The 1x1 convolution layer is built using conv2d_batchnorm with the following
inputs:
  * a. input_layer – the input which is just the previous layer
  * b. filters – number of features to extract
  * c. kernel_size – set to 1
  * d. strides – set to 1


4. output layer:
Output layer does the final classification for each pixel to the appropriate class. This is a 1x1
convolution layer with our 3 classifications classes. A softmax is applied to the results in
order to obtain resulting probability.


### Neural network parameters

The Hyperparameters of this FCN are:

* Learning Rate
* Batch Size
* Epochs
* Steps per Epoch
* Validation Steps

#### Learning Rate

The value used by NN to determine how quickly the weights are adjusted.I experimented with values between 0,01 and 0,00001. Otherwise to much instability is introduced into the network.

I ended up with a learning rate of 0,0005

#### Batch Size

So save memory, not all training input put into the network in one run. Input is divided into subsets called batches. The input is randomly shuffled an then put into the batches.

I ended up with a batch size of 40

#### Epochs

An epoch forward + backward propagation of the entire dataset thought a neural network. Performing this multiple times increases network accuracy without extra training data. The accuracy gain will cease over time, though.
When experimenting with different network architectures I set number of epochs to 10 (this is very low number)  to roughly estimate accuracy and save time. For final run I set number of epochs to 100.
* P.S. Probably it not the best approach. What is the way to "fail fast" when setting network hyperparameters?

#### Steps per Epoch

The steps per epoch is the number of training batches which are passed though network in 1 epoch. There is no need to put every image through the network in every epoch and not putting everything in every time also helps with overfitting.

I ended up with 130 steps per epoch

#### Validation Steps

Number of validation batches which pass the network in 1 epoch. This is the same as steps per epoch with validation images.

I ended up with 50 validation steps

### Summary

**[Rev1]** The accuracy of Neural networks that have been built for this project depends greatly on set of training images used. Accuracy dropped significantly when the hero is far away and misclassifies the hero as ‘other person’, since there is a lack of examples. 
If there was a requirement to track another object (e.g dog, cat, car) instead of a human we would have to provide appropriate training data set, i.e  record enough images of objects we need to track


### Future Enhancements

There are many possible ways to improve the results, here are some examples

1. Use more layers in Decoder and Encoder, make the net deeper or try different architectures.
2. It would be great to learn a way to "fail fast", so each experimentation attempt would not too long
3. Use bigger dataset, for example, using data augmentation.
4. Try different optimizers, e.g Nadam.
