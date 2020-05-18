# Steering Angle Prediction Using ResNet-18
This was the final project completed as a group with Zachary Fisher and Anthony Nguyen for CIS 519 at UPenn.

The goal of this project was to use deep learning methods to accurately predict car steering angle. 

In phase 1 of the project, we began by evaluating various neural network architectures to predict the steering angle of a car using a real-world dataset provided by Udacity, Lyft, and Comma.ai. We found that a custom ResNet-18 architecture had the lowest test RMSE out of all the architectures studied and our score would have placed us in sixth place on the original Udacity leaderboard. We wanted to see how this architecture would actually behave on a car so we used Udacity’s car simulator to test it. In phase 2 of the project, various pre-processing methods were applied to generate eight models using the ResNet-18 architecture. Although we were able to successfully drive the car in the simulator, we found the ResNet-18 architecture may not be best suited for real time steering angle prediction due to its complexity.

## Phase 1: Real World Driving Steering Prediction

### Dataset

The dataset we used for the first half of the project comes from Udacity and Lyft’s Perception Challenge, which can be found [here](https://github.com/udacity/self-driving-car/tree/master/datasets). The training data contains 101,398 images that were recorded using three RGB cameras (labeled left, center, and right) mounted on the vehicle and were time-stamped with the vehicle’s steering wheel angle, motor torque, GPS coordinates, and various sensor data. The test set contained 5614 images labeled with the steering angle.


## Phase 2: Udacity Self-Driving Car Simulator

### Dataset

The second half of the project used training data collected from Udacity’s self-driving car simulator (Udacity, 2016a). The simulator came loaded with two tracks. Track 1 is a relative simple track with mostly simple curves and no hills. Track 2 has a lot more challenging curves and is hilly. Images of these tracks are show in Figure 2. Unfortunately, for unknown reasons, the simulator did not allow us to control the throttle during training data collection; we were forced to collect the steering angle data at full throttle. We drove around the tracks and collected a total of 126,712 images, and the corresponding steering angles, for each track.
