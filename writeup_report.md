# **Behavioral Cloning** 

[//]: # (Image References)

[image1]: ./writeup_images/Loss1.png 
[image2]: ./writeup_images/Loss2.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
	* this is a copy of the file model\_6\_best.h5
* writeup_report.md

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

To start off, I copied the LeNet architecture from the video lectures and initially used this to experiment with the procedure for training the model and running the simulator with it. 

Eventually I used the Nvidia model and modified it to include a number of dropout and max pooling layers. I also increased the size of the dense layers before the output layer. In the end my model had only one dropout layer, situated between the convolutional and dense layers.
 

#### 2. Attempts to reduce overfitting in the model

The model contains only one dropout layer (line 205 of model.py). I kept the default 20% validation split so that 20% of the images generated after data augmentation were used in the validation set. Typical training and validation losses are shown in the figures below for two runs of model training.

![alt text][image1]
![alt text][image2]

The validation loss is typically higher towards the end of training, indicating that some overfitting is present in the model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

Note that in the final model only seven epochs were used for training.

#### 4. Appropriate training data

I generated my own training data for track 1. I used a base of center line driving in the counterclockwise direction and some runs of careful driving around several of the turns. I did not use any recovery data. The data for each training run was stored in separate directories and the generator code was modified so that I could choose to include or not include different training sets.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initially I used the LeNet architecture as suggested in the video. However on the advice of a previous student (Paul Heraty's cheatsheet) I tried the Nvidia model shown in the lecture. I played with dropout and max pooling layers for some time but the final model had only one dropout and no max pooling layers. I feel like additional dropout layers would have made the model more robust (see below).

#### 2. Final Model Architecture

The final model architecture was the Nvidia model with one dropout layer between the convolutional layer and dense layers. I used three dense layers of sizes 512, 128, and 32 neurons before the final output layer.

#### 3. Creation of the Training Set & Training Process

I tried a number of strategies for acquiring training data. Initially I drove the car around the track at top speed since I didn't have to control the throttle. This did not produce good training data since the car spends very little time on the two sharp turns. These turned out to be the most difficult parts of the track for the network to negotiate and the lack of training data for these turns produced a network that couldn't get around the turns.

So I decided to drive carefully around the track for three laps, taking care to slow down on the turns to get plenty of data there. This was the basis of the training data. However most of my models still could not negotiate either of the sharp turns even after extended training (more than 10 epochs). I included more training runs for each turn until the models were able to get around each one. Sometimes including a extra training set for turn 3 would cause a previously successful model model to crash on turn 1. 

I also tried various strategies to help the car recover, for example, training runs where the car veered back and forth while staying on the road. Unfortunately these did not seem to help the network be a better driver and so were not included in the final training sets.

What seemed to be important however was the inclusion of the left and right camera images. The steering correction factor had to be fine tuned through experimentation and I believe this is the reason why I was able to come up with the a fairly successful model without the use of recovery training data. The final steering correction factor was chosen to be five degrees (i.e., plus/minus 0.2). Initially I thought this was going to be far too large since a five degree steering correction corresponds to a very tight turn. However runs with smaller correction factors (one or two degrees) did not produce successful models.

Image flipping was also used but it's not clear to me what influence this had on the final results.

In the end I was using only about 4300 frames of data, or about 26,000 images total, as input to the model.  Eighty percent of these were used for training and 20% for validation. As per the video suggestion, each image was cropped and normalized in a Keras lambda layer.

#### 4. Analysis of the Results

I have to say that construction of the model and acquiring training data was a rather painstaking trial and error process during which I did not keep very careful notes. Hence once I found a model which successfully drove around the track I decided to stop developing it and write up this report. 

Since the training and validation splits are randomly generated and the data is shuffled before training, every time model.py is run the neural network it produces will be slightly different. I would consider the combination of model and training data set to be robust if this process produced a successful driver almost all the time, say at a 95% rate. To test this I generated 10 drivers using the same model and training sets (these are the files "model\_X_...py"). Of these only six were able to drive the car around the track without crashing. In addition, of the six successful models only a few seemed to be fairly good drivers. For example model 3 nearly crashes into the bridge guardrail (see model\_3\_ok.mp4). Notably, running model three again through the simulator did in fact produce a crash (model\_3\_crash.mp4). This indicates that the simulator itself is not deterministic, perhaps there is a built in randomness to the car's initial state, the simulation algorithm has a sensitivity to roundoff, or there is another source of noise.

My conclusion is that this combination of model and training data is not robust and I in fact simply got lucky when the first network made it around the track successfully.

An additional test is to let the best model drive around the track indefinitely and see if it eventually crashes. I let my best model loose in this manner and it did eventually crash on turn 3 after completing perhaps 10 or 15 laps.