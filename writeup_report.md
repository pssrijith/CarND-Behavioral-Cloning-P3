#**Behavioral Cloning** 
    
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[sample_center_image]: ./examples/sample_centre_image.png "Sample Center Image"
[cropped_image]: ./examples/cropped_image.png "Cropped Image"
[aug_left_right]: ./examples/augment_left_center_right.png "Left Center Right camera images"
[aug_flipped]: ./examples/augment_flipped.png "Flipped camera images"
[visualize_loss]: ./examples/visualize_loss.png "Loss Visualization"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive_data_helper/drive_data.py - helper methods to load and create generators from drive data
* net_arch/nvid_arch.py - contains the keras nvidia convnet architecture implementation
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* final_run.mp4 - video of the final model run
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file along with the drive_data_helper/drive_data.py contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The final model that I chose was based on the nvidia convnet architecture paper - (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

My model consists of a convolution neural network with 3 5x5 convoluations followed 2 3x3 convolutions with filter depths between 24 and 64 (net_arch/nvid.arch.py lines 20-27). 
The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer (code line 19).


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting ( code lines 29,31,33). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 28).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the center of the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I cropped the image so that it removes unwanted parts of the image that do not lend to training the model.

Original image:<br/>
![original image][sample_center_image]
<br/>
Cropped image:<br/>
![cropped image][cropped_image]
<br/>

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to come up with a model that keeps the car within the road but avoid overfitting the predominantly curved(to the left) roads on track1. 

In order to gauge how well my model works, I split my image and steering angle data into training, validation and test set (drive_data_helper/drive_data.py code lines 16-19). 

My first step was to use a convolution neural network model similar to the Lenet architecture to see how well the network does on the dataset. The model seemed to have higher bias and the car would just run a few yards before going off track. Then I augmented the data with data from left and right cameras and also added in flipped images to give more training data. I also trained the car by reversing and turning the other way so as to double the training dataset. 

I found that the Lenet model had a high bias (both training and validation losses were high). With all the additional dataset the car would still run off thh road after weaving around for 10 secs.

Then I worked on the nvidia convnet arch model(described in detail below) and the intial model without dropout had a slightly higher mean squared error on the validation set thatn the traning set for 10 epochs. But still the model was able to drive without going off the road on track1. The only problem I could see was that while it was keeping the car centered on most parts of the track, the car seemed to be weaving a lot at the start of the track (which was less curved than the rest).  

To combat the overfitting, I modified the model to add Dropouts in the FC layer (When I added dropouts in the conv layer I noticed the model underperforming). I tuned the dropout starting with a keep probability of 0.2 upto 0.5 and found the best performance at 0.5.

The final step was to run the simulator to see how well the car was driving around track one. As before the car kept within the center of the track and there was very minimal weaving at the begining of the track than before.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model that I chose was based on the nvidia convnet architecture paper - (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

The code for the architecture is in the net_arch/nvid.arch.py file. 

My model consists of a convolution neural network with the following layers

[ input (160x320x3) -> crop-> 78x320x3 -> normalization -> c1:5x5x24:relu -> max(2x2) -> 
c2:5x5x36:relu -> max(2,2) ->  c3:5x55x48:relu ->max(2,2) -> c4: 3x3x64  -> c5 3x3x64 -> 
flat -> dropout ->  fc(120) -> dropout ->  fc(84) -> dropout -> fc(1) ]
    
1. Input Layer: 160x320x3  (RGB image 160 pixels high and 320 pixels wide) 

2. Cropping Layer: crops image to 78x320x3 to remove unwanted parts out of the image. (code line 18)

3. Normalization layer : a Keras Lambda layer that normalizes the values in the range -0.5 to + 0.5 (code line 19)

4. This is followed by three 5x5 convolution layers each followed by a max pooling layer with stride 2. The depth size in the 5x5 layers are 24,36 and 48 respectively (code lines 20-25)

5. Then we have 2 3x3 convolution layers with depth size 64 (code lines 26-27)

6. All the convolution layers above use Relu for non-linearity

7. This is followed by a flatten layer and 2 fully connected layers of sizes 120 (code lines 28-32)

8. finally the output layer which outputs a single steering measurement.(code line 34)


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![sample center image][sample_center_image]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when the go to the edges.

Then I trained the car by turning around and driving the other way to get more data points.

To augment the data sat, I also flipped images and angles as this would give data from a mirror image perspective.e.g., +1 angle to turn on a right curve would become -1 to turn on a left curve. 

Here is an image of car from its 3 cameras following by their flippedimages:

left, center, right camera images:
![left, center and right camera][aug_left_right]

their flipped image:
![flipped][aug_flipped]


After the collection process, I had 21,954 data points. I randomly shuffled the data set and put 10% of the data into a test set and the remaining 90% was split 80:20 into test and validation sets. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by overfitting when I go above that. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![loss visualization][visualize_loss]

My final test score loss was 0.0156
