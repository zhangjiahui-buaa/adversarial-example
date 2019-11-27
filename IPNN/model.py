import torch
import torch.nn as nn
from torch import optim
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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


data_tf = transforms.Compose(
    [transforms.ToTensor()])

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=False)
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=False)

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

net = net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), 1e-1)

nums_epoch = 20

losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for img, label in train_data:

        img,label = img.cuda(),label.cuda()


        _,out = net(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0

    for img, label in test_data:


        img,label = img.cuda(),label.cuda()
        _,out = net(img)

        loss = criterion(out, label)

        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    print('Epoch {} Train Loss {} Train  Accuracy {} Teat Loss {} Test Accuracy {}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data)))

torch.save(net.state_dict(),'model.pt')