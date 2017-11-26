## Follow me project writeup

### Environment:

[gtx]:./images/GTX.jpg
![gpu][gtx]

For this project I decided to build my own GPU platform to experiment with.
Besides my main graphic card, I got extra NVIDIA GTX980 
setting up cuda on Ubuntu 14.04 deserves separate writeup:), but once i got it working, I gained 
more than x10 boost in calculation speed.

### Network architectures

[2lnn]:./images/2_layers_nn.jpg
[3lnn]:./images/3_layers_nn.jpg

##### 2-Layer deep architecture

In this project I experimented with 2 neural network architectures: 2-layer (2 encoders + 2 decoders) and 3-layer (3 encoders + 3 decoders);
Layer sizing is shown on the pictures below

![nn 1][2lnn]


##### 3-Layer deep architecture

![nn 2][3lnn]


### Project components

1. encoder_block(input_layer, filters, strides):
Function builds a convolution layer. This layer is learning learn features with spatial
information by sliding at a window with a certain kernel size. 

a. input_layer - input from previous layer 
b. filters – amount of features that will be extracted 
c. strides – number of strides step we slide the window.

2. decoder_block(small_ip_layer, large_ip_layer, filters):
This is essentially a reverse of the encoder layer. The purpose of this layer is to
upscale the image back to its original size and extract the encoded information the net found
using the encoder blocks. 

a. small_ip_layer – input from the previous layer
b. large_ip_layer – the skip layer
c. filters – amount of features that are extracted


3. 1x1 conv layer. 
This layer extracts non linear features for each pixel in a
layer, and acts basically like FCN (fully connected networn) for each pixel. Like in the encoder layers, the weight
sharing gives allot of pixels for our network to train on no matter where on the image
they reside. The 1x1 convolution layer is built using conv2d_batchnorm with the following
inputs:
    a. input_layer – the input which is just the previous layer
    b. filters – number of features to extract
    c. kernel_size – set to 1
    d. strides – set to 1

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

##### Learning Rate

The value used by NN to determine how quickly the weights are adjusted.I experimented with values between 0,01 and 0,0001. Otherwise to much instability is introduced into the network.

I ended up with a learning rate of 0,005

#### Batch Size

So save memory, not all training input put into the network in one run. Input is divided into subsets called batches. The input is randomly shuffled an then put into the batches.

I ended up with a batch size of 40

#### Epochs

An epoch forward + backward propagation of the entire dataset thought a neural network. Performing this multiple times increases network accuracy without extra training data. The accuracy gain will cease over time, though.
When experimenting with different network architectures I set number of epochs to 10 (this is very low number)  to roughly estimate accuracy and save time. For final run I set number of epochs to 100.
* P.S. Probably it not the best approach. What is the way to "fail fast" when setting network hyperparameters?

#### Steps per Epoch

The steps per epoch is the number of training batches which are passed though network in 1 epoch. There is no need to put every image through the network in every epoch and not putting everything in every time also helps with overfitting.

I ended up with 100 steps per epoch

#### Validation Steps

Number of validation batches which pass the network in 1 epoch. This is the same as steps per epoch with validation images.

I ended up with 50 validation steps

The tuning was a kind of educated brute-force. I started with the default parameters and kept adjusting the learning rate. I then tried various epochs and started varying the steps per epoch. My best score to date is 0.444166837977% score using only the provided training images.


### Future Enhancements

There are many possible ways to improve the results, here are some examples

1. Use more layers in Decoder and Encoder, make net deeper or try different architectures.
2. Use bigger dataset, for example, using data augmentation.
3. Try different optimizers for example Nadam.
