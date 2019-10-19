import copy

import torchvision.datasets as dataset
import torch
import torchvision.transforms as transforms
from de import DE
from de import candidate
import numpy as np
import torchvision
import torch.nn as nn
from train import VGG16
import matplotlib.pyplot as plt
plt.switch_backend('agg')
def generate(image,label):
    parent = [candidate(int(np.random.uniform(0, 32)),
                          int(np.random.uniform(0, 32)),
                          np.random.normal(128/256, 127/256),
                          np.random.normal(128/256, 127/256),
                          np.random.normal(128/256, 127/256)
                          ) for i in range(400)]
    iter_time = 1

    for i in range(iter_time):
        parent,child = DE(parent)
        for j in range(len(parent)):
            if score(image,label,parent[j]) > score(image,label,child[j]):
                parent[j] = child[j]

    final_index = 0
    for i in range(len(parent)):
        if score(image,label,parent[i]) < score(image,label,parent[final_index]):
            final_index = i


    final_candi = parent[final_index]
    final_image = copy.deepcopy(image)
    final_x,final_y = final_candi.get_loc()
    final_R,final_G,final_B = final_candi.get_perturbation()
    final_image[0][0][final_x][final_y] = final_R
    final_image[0][1][final_x][final_y] = final_G
    final_image[0][2][final_x][final_y] = final_B
    return final_image,final_candi

def score(image,label,candi): ## image is (1,3,32,32), candi is (x,y,R,G,B)
    x,y = candi.get_loc()
    if (x>31 or y > 31 or x<0 or y<0 ):
        return float("inf")
    R,G,B = candi.get_perturbation()
    '''if(image[0][0][x][y] + R > 1 or image[0][0][x][y] + R < 0 or
            image[0][1][x][y] + G > 1 or image[0][1][x][y] + G < 0 or
            image[0][2][x][y] + B > 1 or image[0][2][x][y] + B < 0):
        return float("inf")'''

    new_image = copy.deepcopy(image)
    new_image[0][0][x][y] = R
    if(new_image[0][0][x][y] > 1):
        new_image[0][0][x][y] = 1
    if(new_image[0][0][x][y] < 0):
        new_image[0][0][x][y] = 0

    new_image[0][1][x][y] = G
    if (new_image[0][1][x][y] > 1):
        new_image[0][1][x][y] = 1
    if (new_image[0][1][x][y] < 0):
        new_image[0][1][x][y] = 0

    new_image[0][2][x][y] = B
    if (new_image[0][2][x][y] > 1):
        new_image[0][2][x][y] = 1
    if (new_image[0][2][x][y] < 0):
        new_image[0][2][x][y] = 0

    output = model(new_image)
    f = nn.Softmax()
    output = f(output)
    score = output[0][label]
    return score





model = VGG16()
model.load_state_dict(torch.load('model.pt'))
model.eval()
classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

testset = dataset.CIFAR10(root='../dataset', train=False,download=True,transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=1)

if __name__ == '__main__':


    i=0
    final_img_list= []
    for image,label in testloader:

        i += 1
        if i==1:
            break
    final_image,final_candi = generate(image,label)

    x,y = final_candi.get_loc()
    '''if final_image[0][0][x][y] > 1:
        final_image[0][0][x][y] = 1
    if final_image[0][0][x][y] < 0:
        final_image[0][0][x][y] = 0
    
    if final_image[0][1][x][y] > 1:
        final_image[0][1][x][y] = 1
    if final_image[0][1][x][y] < 0:
        final_image[0][1][x][y] = 0
    
    if final_image[0][2][x][y] > 1:
        final_image[0][2][x][y] = 1
    if final_image[0][2][x][y] < 0:
        final_image[0][2][x][y] = 0'''


    img = image[0]
    img = img.numpy()
    img = np.transpose(img, (1,2,0))
    img.tofile('img.bin')
    plt.figure()
    plt.imshow(img)

    output = model(image)
    f = nn.Softmax()
    output = f(output)
    plt.title('origin prediction is :' + classes[label]+ ' with ' + str(output[0][label].item()) + ' confidence')
    plt.show()
    plt.savefig('origin.png')

    final_img = final_image[0]
    final_img = final_img.numpy()
    final_img = np.transpose(final_img, (1,2,0))
    final_out = model(final_image)
    _, predicted = torch.max(final_out, 1)
    final_out = f(final_out)

    final_img.tofile('final_img.bin')
    plt.figure()
    plt.imshow(final_img)
    plt.title('new prediction is :' + classes[predicted] + ' with ' + str(final_out[0][predicted.item()].item()) + ' confidence')
    plt.show()
    plt.savefig('new.png')
    print(final_candi.get_perturbation())
