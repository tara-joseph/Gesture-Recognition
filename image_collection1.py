# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:28:05 2022

@author: Dr. Sony
"""
#===========================================================================================================
#                         PROJECT 2 - HAND GESTURE RECOGNITION
#=========================================================================================================

# Importing required libraries    
    

import cv2
import os
import time
import uuid

#======================================================================================================================

# Making the file directories for the gesture images to be stored

root_directory = "Images"
dataset_directory = ["train", "validate"]

try:
    os.mkdir("Images")
except FileExistsError as f:
     pass

root_path = os.path.join(os.getcwd(), root_directory)
os.chdir(root_path)

for dataset in dataset_directory:
    data_path = os.path.join(root_path, dataset)
    try:
         os.mkdir(data_path)
    except FileExistsError:
        continue
    
#==========================================================================================================
#                      Capturing images for the train data set
#================================================================================================================


images_path = 'C:\\Users\\Dr. Sony\\Images\\train'

labels =['Play','Pause','FF','Rewind','Restart','None']
number_imgs = 300


# Dimensions of the frame
upper_left = (50, 50)
bottom_right = (250, 300)


for label in labels :
    os.makedirs('C:\\Users\\Dr. Sony\\Images\\train\\' +label)
    
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(10)
    
    for imgnum in range(number_imgs):
        ret,frame = cap.read()
        
        # Rectangle box
        r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 4)
        roi_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        roi_img = cv2.resize(roi_img, (224, 224)) 
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
 
        
        # Region of Interest
        cv2.imshow("Live capturing",frame)
        key = cv2.waitKey(1)
        
        imagename = os.path.join(images_path,label, label+ '.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,roi_img)
        cv2.imshow('frame',roi_img)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()       
cv2.destroyAllWindows()

#================================================================================================================