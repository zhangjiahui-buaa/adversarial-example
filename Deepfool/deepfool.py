import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import copy
import numpy as np
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import matplotlib.pyplot as plt
import train

bt = 8
net = train.VGG16()
net.load_state_dict(torch.load('../model/model.pt'))
transform = transforms.Compose([ 
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),                         #将image转化为tensor
    transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                         ( 0.229, 0.224, 0.225 ))])#三个通道的均值和方差
testset = datasets.ImageNet(root='../dataset', split = 'val', transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size = bt,shuffle = True,num_workers = 12)

#testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
classes = 1000
categories = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def showimage(tensor):
    im = tensor.detach()[0].numpy()
    im = np.transpose(im,(1,2,0))
    plt.imshow(im)
def df(image, f, classes = 1000,overshoot=0.02, max_loop=50):  # x is an image of size bt_size(1) * 3 * 32 * 32, f is the model(vgg16)

    x = copy.deepcopy(image)
    x.requires_grad_()

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)



    out = f(x)
    _, label = torch.max(out, 1)
    label = label.item()
    k_i = label
    loop = 0

    while k_i == label and loop < max_loop:
        zero_gradients(x)
        out[0,label].backward(retain_graph=True)
        ori_grad = x.grad.data.cpu().numpy().copy()
        pert = np.inf

        for k in range(classes):
            if k == label:
                continue
            zero_gradients(x)
            out[0,k].backward(retain_graph=True)

            cur_grad = x.grad.data.cpu().numpy().copy()
            w_k = cur_grad - ori_grad
            f_k = out[0,k].item() - out[0,label].item()
            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten(), ord=2)
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = ((pert+1e-4) * w)
        r_i = r_i/np.linalg.norm(w.flatten(),ord=2)

        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot)

        x = pert_image

        x.detach_()
        x.requires_grad_()
        out = f(x)
        _, k_i = torch.max(out, 1)
        k_i = k_i.item()
        loop += 1
    r_tot = (1 + overshoot) * r_tot
    return r_tot, loop, label, k_i, pert_image

def FGSM(model, criteria, original_input, true_label, pertubation):
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()
    output = model(x_star)
    loss = criteria(output,true_label)
    loss.backward()
    #print(x_star.grad)
    g = x_star.grad
    x_star = x_star + pertubation*torch.sign(g)
    return x_star
if __name__ == '__main__':
    loss = nn.CrossEntropyLoss()
    success = 0
    s = 0
    total = 0
    dis = 0
    for image,label in testloader:
        #fun(image,net)
        #image,label = image.cuda(),label.cuda()
        '''r_tot,loop,ori_label,k_i,pert_image = df(image,net)
        dis += np.linalg.norm((pert_image-image).detach().numpy().flatten(),ord = np.inf)

        if(ori_label!=k_i):
            success +=1
        total += 1
        x_star = FGSM(net,loss,image,label,10/256)
        _,pred = torch.max(net(x_star),1)
        if(pred!=label):
            s+=1
        #print(total)
        if total == 500:
            print("success rate of Deepfool: "+str(success/total))
            print("success rate of FGSM: " + str(s/total))
            print("average distance " + str(dis*256/total))
            break'''
        break
