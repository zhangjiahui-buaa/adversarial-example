import torch

import attack
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from train import VGG16

if __name__ == '__main__':
    model = VGG16()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    classes = ['plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    testset = dataset.CIFAR10(root='../dataset', train=False,download=True,transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(testset, batch_size=1)
    f = nn.Softmax()
    success = 0
    total = 0
    for image,label in testloader:
        output = model(image)
        _,pred = torch.max(output,1)
        if(label == pred):
            output = f(output)
            print('origin prediction is '+ classes[label] + ' with '+ str(output[0][label].item())+' confidence')

            final_image,final_candi = attack.generate(image,label)

            x,y = final_candi.get_loc()
            if(final_image[0][0][x][y] > 1):
                final_image[0][0][x][y] =1
            if (final_image[0][0][x][y] < 0):
                final_image[0][0][x][y] =0

            if (final_image[0][1][x][y] > 1):
                final_image[0][1][x][y] = 1
            if (final_image[0][1][x][y] < 0):
                final_image[0][1][x][y] = 0

            if (final_image[0][2][x][y] > 1):
                final_image[0][2][x][y] = 1
            if (final_image[0][2][x][y] < 0):
                final_image[0][2][x][y] = 0

            new_out = model(final_image)
            _,new_pred = torch.max(new_out,1)
            new_out = f(new_out)
            print('new prediction is '+ classes[new_pred.item()] + ' with '+ str(new_out[0][new_pred.item()].item())+' confidence')

            if(new_pred!=pred):
                success +=1
            total+=1
            if total==2:
                break

    print(success/total)




