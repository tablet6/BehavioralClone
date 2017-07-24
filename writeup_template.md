#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia.png "Model Visualization"
[image2]: ./examples/image-bgr-left.png "Left"
[image3]: ./examples/image-bgr-center.png "Center"
[image4]: ./examples/image-bgr-right.png "Right"
[image5]: ./examples/image-rgb-center.png "RGB center"
[image6]: ./examples/image-rgb-center-flip.png "RGB center Flip"
[image7]: ./examples/image-bright.png "Light"
[image8]: ./examples/center_track2-1.jpg "Track2 1"
[image9]: ./examples/center_track2-2.jpg "Track2 2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md  summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses nVidia's CNN Architecture. 

Lines 82-94

* First Layer: Image normalization using a Keras lambda layer.
* Layers 2-4: Convolutions with a 2x2 stride and a 5x5 kernel size, with relu activation for non-linearity.
* Layers 5-6: Convolutions with a 3×3 kernel size with no stride, with relu activation.
* Following the above layers with three fully connected layers, leading to a final output control value.

####2. Attempts to reduce overfitting in the model

My nVidia model initially used Dropout layers to avoid overfitting but turned out it was bug with my correction values of left and right images. At the end I didn't need to use any Dropout layer, and went with the nVidia CNN architecture.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 32-69). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 97).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the center of the road. I used:

* 2-3 laps of center lane driving of Track 1.
* One Recovery loop driving of Track 1.
* 2 laps of center driving of Track 2.
* Some focussed Recovery driving from Track 2's first sharp curve.

A total of 58K images.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use the same CNN model similar to the Class on Udacity Lesson: Behavioral Cloning. I followed a model which had normalization, Cropping, Convolution layer, MaxPooling, Convolution Layer, Max Pooling and 3 Fully connected layers with RELU activation for non-linearity. I introduced couple of Dropout layers when the model was overfitting. I wasn't getting good convergence.

I shifted to the nVidia architecure (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
which uses and operates on the same style of collecting images with left, center and right camera. This model is more appropriate for this project because it detects features only with the steering angle and moreover, it is an end-to-end approach which is powerful. 

I split my data into training and validation set (clone.py, line: 16) to determine how the model is working. Initially I had both training and validation with very low mse (in the range of 0.0008). Since both are very low, I collected more Recovery data. But the car fails to pass the first curve after the bridge. Later I realized it was a bug in using correction values for left and center images. Fixing the correction values, I see the mse is better and not overfitting. 

My training was going really slow at this point with 18K images. So I shifted to using a generator, which dramatically increased the speed of training.

I still saw the car sometimes fails to complete a complex curve or touches the curb. To improve the training, I increased samplesPerEpoch, which fixed those issues.

I let the car run for a long time, and sometimes, I see the car is going offtrack after 15 or 20 laps.

I then moved on to test the Track 2 with this model without collecting any of its data. It started of surprisingly well, but couldn't do a complex turn. I collected a lap of data and some recovery data for the turn. My sample images went from 28K to 38K. After training, I still see the car not turning the complex curve.

I went ahead and collected another 1.5 laps of Track 2 with some recovery data. The sample data ballooned to 58K.

Adjusting the samplesPerEpoch to 48000, the car was able to complete Track2 successfully.

I then re-ran Track 1, and the car kept running fine for over 3 hours autonomously.


####2. Final Model Architecture

The final model architecture (clone.py lines 82-94) consisted of a convolution neural network with the following layers, which uses nVidia CNN architecture.

It has these following layers:
Lines 82-94

* First Layer: Image normalization using a Keras lambda layer.
* Layers 2-4: Convolutions with a 2x2 stride and a 5x5 kernel size, with relu activation for non-linearity.
* Layers 5-6: Convolutions with a 3×3 kernel size with no stride, with relu activation.
* Following the above layers with three fully connected layers, leading to a final output control value.


Here is a visualization of the architecture.
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Each captures, left, center and right images.

Here are the 3 images of left, center and right images.

![alt text][image2]
![alt text][image3]
![alt text][image4]


I converted these to RGB images. Here is a center RGB image.

![alt text][image5]


I flipped these images to simulate reverse driving on the same Track. 
This helps with more data points to the model instead of it memorizing the same lane.
Here is a flipped image of above center image.

![alt text][image6]


I modified the brightness of the above image, to train the model to drive in different daylight situations. Here is a brightness adjusted of the above image.

![alt text][image7]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover.

After the Track 1 was behaving decently, I tested my model on Track 2, which failed on a sharp turn.
Then I repeated the data collection process on track two in order to get more data points. To augment the data set, I  converted to RGB, Flipped and adjusted Brightness of each sample image.

Here are examples of Track 2 images.

![alt text][image8]
![alt text][image9]

After the collection process, I had 58K data points. 
I randomly shuffled the data set and put 20% of the data into a validation set. 
The validation set helped determine if the model was over or under fitting. 

I used 3 Epochs and I used an adam optimizer so that manually training the learning rate wasn't necessary. I added a correction of 0.2, -0.2 to left and right images.

This is my output of the model:

* samplesPerEpoch:  48000
* validationSamples:  3899
* batch_size:  32
* Epoch 1/3
	- 611s - loss: 0.0905 - val_loss: 0.0780
* Epoch 2/3
	- 599s - loss: 0.0698 - val_loss: 0.0668
* Epoch 3/3
	- 605s - loss: 0.0647 - val_loss: 0.0667


### Here are the videos of autonomously driven Track 1 and Track 2.

[![image alt text](https://img.youtube.com/vi/l4jD8i4OPF4/0.jpg)]
(https://youtu.be/l4jD8i4OPF4)

[![image alt text](https://img.youtube.com/vi/JJeLU1T4LS8/0.jpg)]
(https://www.youtube.com/watch?v=JJeLU1T4LS8)

### Videos from 3rd Person view of Track1 and Track2.

[![image alt text](https://img.youtube.com/vi/5RkqrEciicI/0.jpg)]
(https://www.youtube.com/watch?v=5RkqrEciicI)

[![image alt text](https://img.youtube.com/vi/uDm_BW0l9wg/0.jpg)]
(https://www.youtube.com/watch?v=uDm_BW0l9wg)



### Improvements
- When using Fantasic Graphics Quality, where shadows are more prominent, my model is treating those as obstacles and tries to avoid it. Doing that the car is going off-track. I need to augument my data set with shadow images

- My model on Track 1 behaves well with 8 to 30 mph. But on Track 2, it goes offtrack at 30mph and shadows.
