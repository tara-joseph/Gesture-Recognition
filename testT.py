# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:32:07 2022

@author: Dr. Sony
"""


#---------------------------------------------------------------------------------
# TESTING

# Importing necessary modules

import cv2
import torch
from PIL import Image
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

gesture_names = ['FF','None','Pause','Play','Restart','Rewind']

# Loading the stored model

alex_model = torch.load("alex2.pth")
alex_model.eval()


# Image transformations


upper_left = (50, 50)
bottom_right = (250, 300)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# Opening the webcam

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# Creating a window with a rectangular frame 

    cv2.namedWindow("Gesture Recognition Cam", cv2.WINDOW_NORMAL)
    
     # Rectangle box
    r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 4)
    roi_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    roi_img = cv2.resize(roi_img, (224, 224)) 
    #roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    image = transform(Image.fromarray(roi_img))
    image = image.view(1, 3, 224, 224)

    pred = alex_model(image)
    gesture_pred = int(torch.max(pred.data, 1)[1].numpy())

    cv2.putText(frame, f"The predicted gesture is {gesture_names[gesture_pred]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition Cam", frame)
cap.release()
cv2.destroyAllWindows()

#======================================================================================================
