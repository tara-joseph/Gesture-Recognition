# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:43:53 2022

@author: Dr. Sony
"""

#===========================================================================================================
#                         PROJECT 2 - HAND GESTURE RECOGNITION
#=========================================================================================================

#                            TRAINING MODEL - VGG16
#==========================================================================================================

# Importing required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion() 

#==========================================================================================================

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#=============================================================================================================

# Directory path where the images are stored
data_dir = 'C:\\Users\\Dr. Sony\\Images'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validate']}

# Loading the images
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validate']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}


class_names = image_datasets['train'].classes
print(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#=============================================================================================================================

# Defining a function to train the model

def train_model(model, criterion, optimizer,scheduler,num_epochs=3): #check with scheduler
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #if phase == 'train':
                #scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'validate' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#================================================================================================================

   
# Training the VGG MODEL

model_vgg = models.vgg16(pretrained=True)

model_vgg.eval() 


model_vgg.classifier[6] = nn.Linear(4096,6)

model_vgg = model_vgg.to(device)


#for param in model_ft.parameters():
  #param.requires_grad = False

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_vgg.parameters(), lr = 0.001 , momentum = 0.9)

# Decay LR by a factor of 0.1 every 3 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model_vgg = train_model(model_vgg, criterion, optimizer_ft,exp_lr_scheduler,
                       num_epochs = 3)

""" Epoch 0/2
----------
train Loss: 0.6670 Acc: 0.7558
validate Loss: 0.0677 Acc: 0.9933

Epoch 1/2
----------
train Loss: 0.3592 Acc: 0.8779
validate Loss: 0.0069 Acc: 1.0000

Epoch 2/2
----------
train Loss: 0.2825 Acc: 0.9073
validate Loss: 0.0162 Acc: 1.0000

Training complete in 366m 50s
Best val Acc: 1.000000"""

#==========================================================================================================

# Creating a confusion matrix


nb_classes = 6

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['validate']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_vgg(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

print(confusion_matrix.diag()/confusion_matrix.sum(1))


# saving the model 

torch.save(model_vgg, "gesture_model_vgg.pth")    
