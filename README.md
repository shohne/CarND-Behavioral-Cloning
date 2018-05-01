# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repo contains the code written to complete the project **Behaviorial Cloning** on Udacity Self-Driving Car Nanodegree. The goal is train a deep learning model able to control (steer) a car and drive it safely on track.

Prerequisites
---
To run this project, you need [Anaconda 4.5.1](https://anaconda.org/conda-canary/conda/files?version=4.5.1) installed.

Installation
---
First, clone the repository:
```
git clone https://github.com/shohne/CarND-Behavioral-Cloning.git
```
Change current directory:
```
cd CarND-Behavioral-Cloning
```
Create the conda environment with all dependencies:
```
conda env create -f environment.yml
```
The name of created environment is *carnd-behaviorial-cloning-hohne*.

Activate the created conda environment:
```
source activate carnd-behaviorial-cloning-hohne
```
Running the Simulator
---
File **model.h5** has a pretrained keras model that could be used to control steer when driving. To use it, first launch [Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) in  **Autonomous Mode** and select track 1. Now, in terminal, execute python program that send control signals to the simulator:
```
python drive.py model.h5
```
The controller should be able to drive for a long time. A recorded video of this execution could be seen in:

[Track 1](video_track_1.mp3)

This model is able to drive on track 2 (at reduced speed) too. To test it, just choose **Autonomous Mode**, **track 2** and execute the controller:
```
python drive.py model.h5 25
```

[Track 2](video_track_2.mp3)

List of Main Files
---
* **drive.py** python program responsible to receive images from the simulator and predict steering angle to keep car on track;
* **model.py** python script that build keras neural model and train it;
* **model.h5** pretrained keras model (using *model.py*);
* **video_1.mp3** recorded video of simulator in **Autonomous Mode** driving on track 1 and using given *model.h5*;
* **video_2.mp3** recorded video of simulator in **Autonomous Mode** driving on track 2 and using given *model.h5*;
* **data** (directory) must contain dataset file **driving_log.csv** and **IMG** folder with captured images and labels;
* **environment.yml** python dependencies;
* **README.md**;
* [**report.md**](report.md).

Train and Implementation Details
---
To know how [model.h5](model.h5) was trained and implementation details, please visit [report](report.md).
