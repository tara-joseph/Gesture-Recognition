# Gesture-Recognition
The proposed system gives a recognition algorithm which recognize a set of six specific static hand gestures, namely, Pause, Play, Rewind, Restart, FF(Fast Forward), and None.

# Problem statement
The problem statement at hand is to recognize different hand gestures that are performed over a
real time webcam stream. The gesture images that are captured and stored are used to train and
build deep learning models. A real-time stream will be used to assess the model's ability to
recognize gestures

# Procedure
1.Data Preparation - The first stage consists of capturing the images by opening the webcam and saving the
captured images to the respective folders.

2.Model Training -  makes use of the PyTorch library to
make any data modifications and to load the pretrained models through transfer learning.This
stage also consists of benchmarking the performance of each of the models and choosing the best
model for testing.

3.Model Testing - the gestures are performed again in real time where it is
predicted.

