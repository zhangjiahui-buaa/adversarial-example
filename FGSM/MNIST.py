import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(in_features=size,out_features=classes)
    def forward(self, x):
        x = self.fc(x)
        return x


size = 28*28
classes = 10
bz = 64
lr = 0.5
epochs = 100
train_set = dataset.MNIST(root='../../Dataset',
                          train = True,
                          transform= transforms.ToTensor(),
                          download = False)
test_set = dataset.MNIST(root = '../../Dataset',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=False)
train_loader = torch.utils.data.DataLoader(dataset = train_set,batch_size = bz,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size = bz,shuffle = True)


model = net()

critera = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
schedule = torch.optim.lr_scheduler.StepLR(optimizer,40)

for epoch in range(epochs):
    i = 0
    for images,labels in train_loader:
        if i == 0:
            print(labels.shape)
        images = images.reshape(-1,size)
        outputs = model(images)
        if i == 0:
            print(outputs.shape)
        _,predicted = torch.max(outputs.data,1)
        total = labels.size(0)
        corrected = (predicted==labels).sum().item()
        acc = corrected/total
        loss = critera(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if i%100 == 0:
            print("epoch [{}/{}],iter[{}],acc{:.4f},loss{:.4f}".format(epoch+1,epochs,i,acc,loss))
        i += 1
torch.save(model.state_dict(), 'model.pt')