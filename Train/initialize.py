from PIL import Image
import numpy as np
from torchvision import transforms 
import torchvision
import torch
import torch.optim as optim
import os,glob

trafo=transforms.ToTensor()


def load_data(batch_ang, batch_vel, iftrain): #iftrain = True:traindata; False = testdata
    x_train=[]
    y_train_vel=[]
    y_train_ang=[]
    data_train = []
    data_train2 = []

    if iftrain == True:
        IMAGE_PATH="/home/axelk/Documents/Data/train/" #You should modify this to point to your train data
    else:
        IMAGE_PATH="/home/axelk/Documents/Data/test/" #and modify this for test data

    for filename in glob.glob(os.path.join(IMAGE_PATH, '*.png')): #Images saved based on scheme seen in preprocess.py
        with open(filename, 'r') as f:

            im=Image.open(filename)
            h=trafo(im)
            x=np.array(h)
            x/=255
            x_train.append(x)
            s = filename
            s=s.replace('png','txt')
            s=s.replace('train_im', 'train_vel')
            velocity=np.loadtxt(str(s))
            s=s.replace('train_vel', 'train_ang')
            angle=np.loadtxt(str(s))
            y_train_vel.append(velocity)
            y_train_ang.append(angle)
    x_train=np.array(x_train)
    y_train_vel = np.array(y_train_vel)
    y_train_ang = np.array(y_train_ang)
    for i in range(0, len(x_train)):
        data_train.append([x_train[i],y_train_vel[i]])
        data_train2.append([x_train[i],y_train_ang[i]])
    print(x_train.shape)
    print(y_train_vel.shape)
    print(y_train_ang.shape)
    
    if iftrain == True:
        trainloader = torch.utils.data.DataLoader(data_train, batch_size=batch_vel,
                                          shuffle=True, num_workers=2)
        trainloader2 = torch.utils.data.DataLoader(data_train2, batch_size=batch_ang,
                                          shuffle=True, num_workers=2)
        return trainloader, trainloader2
    else:
        testloader = torch.utils.data.DataLoader(data_train, batch_size=batch_vel,
                                          shuffle=True, num_workers=2)
        testloader2 = torch.utils.data.DataLoader(data_train2, batch_size=batch_ang,
                                          shuffle=True, num_workers=2)
        return testloader, testloader2
