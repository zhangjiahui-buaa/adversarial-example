import torchvision

import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title):
    npimg = img.cpu().detach().numpy()
    plt.figure(figsize = (10, 20))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

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
                                nn.ReLU())
        self.fc2 = nn.Linear(in_features=128,out_features=10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.fc2(x)
        return x,y


test_set = mnist.MNIST('../../../Dataset/', train=False, transform=transforms.ToTensor(), download=False)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net().to(device)
net.load_state_dict(torch.load('model.pt'))
net.eval()


number = 2 ## decide how many pictures you want
tn = 8     ## choose top-tn pictures that maximize the neuron's value
## show the semantic information that a single neuron consists
for i in range(number):
    iden = torch.eye(128)[i,:].to(device)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        temp,outputs = net(images)
        value = torch.mv(temp,iden)

    idx = np.argsort(value.cpu().detach().numpy())[-tn:]
    img = images[idx]
    imshow(torchvision.utils.make_grid(img, normalize=True), "Pictures that maximize the "+str(i+1) +"th neuron")


## show the semantic information that the union of random neurons consist
for i in range(number):
    iden = torch.rand(128).to(device)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        temp, outputs = net(images)
        value = torch.mv(temp, iden)

    idx = np.argsort(value.cpu().detach().numpy())[-tn:]
    img = images[idx]
    imshow(torchvision.utils.make_grid(img, normalize=True), "Pictures that maximize the union of random neurons")


