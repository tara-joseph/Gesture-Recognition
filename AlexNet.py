# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:26:29 2022

@author: Dr. Sony
"""
#===========================================================================================================
#                         PROJECT 2 - HAND GESTURE RECOGNITION
#=========================================================================================================

#                            TRAINING MODEL - AlexNet
#==========================================================================================================


# Required packages
from _future_ import print_function, division
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

#=========================================================================================================

#Converting to tensors and other transformations
data_transforms = {
    "train": transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ]),
    "validate": transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])}

# Path of the file where images are stored
data_dir = 'C:\\Users\\Dr. Sony\\Images'


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validate']}

#Loading the images
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'validate']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}

# Printing the 6 class names
class_names = image_datasets['train'].classes
print(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#===============================================================================================================

# Defining a model function

def train_model(model, criterion, optimizer,scheduler,num_epochs=3): 
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

#=================================================================================================================================
    
#Pretrained model

model = models.alexnet(pretrained = True)

model.classifier[6] = nn.Linear(4096,6)

# requires_grad=True so only the new layer’s parameters will be updated
for param in model.parameters():
    param.requires_grad = False

for params in model.classifier.parameters():
    params.requires_grad = True
    
    
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr = 0.001 , momentum = 0.9)
# optimizer = torch.optim.Adam(model_an.parameters(), lr = 0.001)

# Decay LR by a factor of 0.1 every 3 epochs #check w 7
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

model= train_model(model, criterion, optimizer_ft,exp_lr_scheduler,
                       num_epochs = 3)    

##Output

"""Epoch 0/2
----------
train Loss: 0.1388 Acc: 0.9534
validate Loss: 0.0478 Acc: 0.9900

Epoch 1/2
----------
train Loss: 0.0186 Acc: 0.9950
validate Loss: 0.0066 Acc: 0.9983

Epoch 2/2
----------
train Loss: 0.0045 Acc: 0.9989
validate Loss: 0.0164 Acc: 0.9950

Training complete in 12m 59s
Best val Acc: 0.998333"""

#==========================================================================================================

# Creating a confusion matrix


nb_classes = 6

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['validate']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

print(confusion_matrix.diag()/confusion_matrix.sum(1))

"""print(confusion_matrix.diag()/confusion_matrix.sum(1))
tensor([1.0000, 0.9800, 1.0000, 0.9900, 0.9400, 0.9900])"""

#======================================================================================================================
# Saving the model trained

torch.save(model, "alex2.pth")

#================================================================================================================