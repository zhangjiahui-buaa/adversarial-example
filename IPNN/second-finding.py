import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
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
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net().to(device)
net.load_state_dict(torch.load('model.pt'))
net.eval()

target_class = torch.tensor([8]).cuda()  ## the class which you want to get

origin_pic = test_set[0][0].reshape(-1,1,28,28).cuda() ## the original picture
_,origin_outputs = net(origin_pic)
_,origin_pred = torch.max(origin_outputs,1)     ## the original prediction

plt.figure()
plt.imshow(origin_pic.cpu().reshape(28,28).numpy(),cmap='gray')
plt.show()

r = torch.rand(1,1,28,28).cuda()
r.requires_grad_()
optimizer = optim.SGD([r],lr = 0.01)
criteria = nn.CrossEntropyLoss()
scheduel = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
iter_time = 10000
c = 1               ## punishment factor, need to adjust to a proper value.

for i in range(iter_time):
    x_r = origin_pic + r
    _,outputs = net(x_r)
    _,pred = torch.max(outputs,1)

    loss = c*r.abs().sum()+criteria(outputs,target_class)
    if i%100==0:
        print(loss)
        scheduel.step(loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()


if pred.item() == target_class:
    print("Attack successed!")
    plt.figure()
    plt.imshow(x_r.cpu().reshape(28,28).detach().numpy(),cmap='gray')
    plt.title('New prediction !:' + str(pred.item()))
    plt.show()



