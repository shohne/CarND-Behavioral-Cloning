# **Behavioral Cloning Report**

The goals of this project are:
1. To build a model to drive the [Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim);
* To define and obtain the appropriate dataset to train the driver model
* To reflect about upon the pros and possible improvements in shown solution.

### Pipeline
The goal is predict the steer to drive [Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim). To perform this task, the driver needs to know the Car state on the road, in fact, Simulator captures and send some information to the controller. The controller has access to:
* current speed;
* current throttle;
* current steering angle;
* current images road taken from 3 cameras inside the Car.

The controller is a program that using these data needs to compute throttle and steering angle. It sends these data and simulator is responsible to compute the next Car state what begins the circle:

**Receive State** Car --> **Predict** throttle and steering --> **Send control** message to simulator

again.

The crucial task is **predict** and to achieve it, we have trained a **Deep Learning Model**. As we known, **Machine Learning** try to *learn* to do a task from a lot of data that shows the expected behavior. To acomplish this, we have drove the simulator in **training mode** capturing relevent information (camera images and steering angle). The idea is that the model could mimic the behavior given by examples.

### Model Architecture
At a rate of 20Hz in training mode, simulator could give us following information:
* Image from center camera. This image focus on the road and is 320 pixels wide versus 160 pixels height, it is 24 bits colorful per pixel image as:

![example image](example_image.jpg)

* Images from left and right camera;
* current speed, throttle and steering angle.

The proposed model uses only the center image and try to predict the steering angle as the diagram:

![Driver Model](model.png)

We have used [keras](https://keras.io/) to define the model. Following table explicit each layer properties:


Layer (type) | Output Shape | Param #   
-------------|--------------|---------
image (InputLayer) | (160, 320, 3) | 0
lambda_1 (Lambda) | (160, 320, 3) | 0
conv_1 (Conv2D) | (160, 320, 24) | 7224
relu_1 (Activation) | (160, 320, 24) | 0
maxpool_1 (MaxPooling2D) | (79, 53, 24) | 0
conv_2 (Conv2D) | (79, 53, 36) | 42372
relu_2 (Activation) | (79, 53, 36) | 0
maxpool_2 (MaxPooling2D) | (39, 8, 36) | 0
conv_3 (Conv2D) | (39, 8, 48) | 43248
relu_3 (Activation) | (39, 8, 48) | 0
maxpool_3 (MaxPooling2D) | (19, 3, 48) | 0
flatten_1 (Flatten) | (2736) | 0
fc1 (Dense) | (512) | 1401344
activation_1 (Activation) | (512) | 0
dropout_fc1 (Dropout) | (512) | 0
fc3 (Dense) | (16) | 8208
predict_steer (Dense) | (1) | 17

Total params: **1,502,413**

### Dataset
To train the model there should be a directory **data** with file **driving_log.csv** and **IMG** folder. The **driving_log.csv** must have following columns data separated by comma (,):

* filename center image;
* filename right image;
* filename left image;
* steer;
* acceleration;
* break;
* speed;

We are using only center image and steer data, then it is OK to have fake data in filename right, left image, acceleration, break and speed. [Dataset file](https://s3-us-west-1.amazonaws.com/carnd-dataset-hohne/dataset_carnd_behavioral_cloning.zip) has the data used to obtain the trained [model.h5](model.h5). It has 46260 datapoints from 5 sources:

* dataset provided by Udacity that contains data captured driving on track 1 ( datapoints);
* driving on track 1 trying to keep the center of the lane ( datapoints);
* driving on track 1 while trying to recover to the center of the lane ( datapoints);
* driving on track 2 trying to keep the center of the lane ( datapoints);
* driving on track 2 while trying to recover to the center of the lane ( datapoints).

### Train Procedure
First, script **model.py** try to load the dataset, if it could not to find it, then it is downloaded from Internet. After, dataset is splitted in:

* train dataset (70% of whole dataset);
* validation dataset (10%);
* test dataset (20%).

After, verify if a pretrained model exists (file [model.h5](model.h5)), if it exists, load it. If there is not a previous model.h5 file, build a new model.

Next step, starts to train the model calling the **get_data_generador** to provide data for each minibatch. We have choose [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) with hyperparameters:

* **learning rate**: 0.0075;
* **decay**: 1e-6;
* **momentum**: 0.9;
* **nesterov**;
* **epochs** 150;

![Loss Train - Loss Validation X Epoch](train_history.png)


### Result
### Conclusion
