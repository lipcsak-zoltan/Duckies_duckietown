import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class veloCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ru=nn.LeakyReLU()
        #input:128x64, 3
        self.conv1=nn.Conv2d(3,32,3,stride=4) # input RGB, 3 channels
        self.conv2=nn.Conv2d(32,64,5,stride=4) 
        self.pad = nn.ZeroPad2d(2)
        self.conv3=nn.Conv2d(64,128,5,stride=2)
        self.conv4=nn.Conv2d(128,64,1,stride=2)
        self.bn1 = nn.BatchNorm2d(32) 
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.dout=nn.Dropout(0.3)
        self.flat= nn.Flatten(1,2)
        self.lin1= nn.Linear(128,32)
        self.lin2=nn.Linear(32,4)
        
    def forward(self, x):
        x = self.bn1((self.ru(self.conv1(x))))
        x = self.pad(self.bn2((self.ru(self.conv2(x)))))
        x = self.bn3(self.ru(self.conv3(x)))
        x = self.bn4(self.ru(self.conv4(x)))
        x = torch.flatten(x,1)
        x = self.ru(self.lin1(x))
        output = self.ru(self.lin2(x))
        return output
   
class angleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ru=nn.LeakyReLU()
        #input:128x64, 3
        self.conv1=nn.Conv2d(3,32,3,stride=4) # input RGB, 3 channels
        self.conv2=nn.Conv2d(32,64,5,stride=4) 
        self.pad = nn.ZeroPad2d(2)
        self.conv3=nn.Conv2d(64,128,5,stride=2)
        self.conv4=nn.Conv2d(128,64,1,stride=2)
        self.bn1 = nn.BatchNorm2d(32) 
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.dout=nn.Dropout(0.3)
        self.flat= nn.Flatten(1,2)
        self.lin1= nn.Linear(128,64)
        self.lin2=nn.Linear(64,3)
        
    def forward(self, x):
        x = self.bn1((self.ru(self.conv1(x))))
        x = self.pad(self.bn2((self.ru(self.conv2(x)))))
        x = self.bn3(self.ru(self.conv3(x)))
        x = self.bn4(self.ru(self.conv4(x)))
        x = torch.flatten(x,1)
        x = self.ru(self.lin1(x))
        output = self.ru(self.lin2(x))
        return output
   