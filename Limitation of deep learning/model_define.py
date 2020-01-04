import torch
import torch.nn as nn
import torchvision
import  numpy as np
import matplotlib.pyplot as plt
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3),#  bz*16*26*26
                                   nn.BatchNorm2d(num_features=16),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3), # bz*32*24*24
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2))# bz*32*12*12
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3), # bz*64*10*10
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2))#bz*64*5*5
        self.fc = nn.Sequential(nn.Linear(in_features=64*5*5,out_features=512),#bz*512
                                nn.ReLU(),
                                nn.Linear(in_features=512,out_features=128),#bz*128
                                nn.ReLU())#bz*10
        self.fc2 = nn.Linear(in_features=128,out_features=10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.fc2(x)
        return x,y

