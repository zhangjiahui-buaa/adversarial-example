import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
epson = 0.2  ## change this variable in order to constrain the adversary's capability. The greater epson is, more capability the adversary gets.
index = 11    ##change this variable in order to get different iamges
size = 28*28
classes = 10

## net definition
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(in_features=size,out_features=classes)
    def forward(self, x):
        x = self.fc(x)
        return x

model = net()
model.load_state_dict(torch.load('model.pt')) ## load pretrained model

## load test_data
test_set = dataset.MNIST(root = './',
                         train=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.1307,),(0.3081,))]),
                         download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size = 1,shuffle = False)
i = 0


for image,label in test_loader:
    if i==index:
        image = image.reshape(-1,784)
        plt.figure()
        plt.imshow(image.reshape(28,28).detach().numpy(),cmap = 'gray')
        image.requires_grad_()
        outputs = model(image)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(outputs, label)
        prob = F.softmax(outputs)[0][label.item()]*100
        _, predicted = torch.max(outputs.data, 1)
        plt.title('original prediction is ' +str(predicted.item())+' with confidence {:.2f}'.format(prob.item()))
        plt.show()
        loss.backward()

        plt.figure()
        pert = epson *image.grad.sign()
        plt.imshow(pert.reshape(28,28).detach().numpy(),cmap = 'gray')
        plt.title('The perturbation')
        plt.show()

        plt.figure()
        new =  image + epson *image.grad.sign()
        plt.imshow(new.reshape(28,28).detach().numpy(),cmap = 'gray')
        new_outputs = model(new.reshape(-1, 784))
        new_loss = criteria(new_outputs, label)
        _, new_pred = torch.max(new_outputs.data, 1)
        new_prob = F.softmax(new_outputs)[0][new_pred.item()]*100
        plt.title('after FGSM,the new prediction is ' + str(new_pred.item())+' with confidence {:.2f}'.format(new_prob.item()))
        plt.show()
        break
    i+=1

