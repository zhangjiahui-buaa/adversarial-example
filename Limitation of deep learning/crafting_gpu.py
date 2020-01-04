import time

import torch
import numpy as np
import model_define
import torchvision.datasets as dataset
import torchvision.transforms as transform

classes = 10


def judge(i):
    if i < 0:
        return 0
    elif i > 1:
        return 1
    return i


def crafting(x, y_star, out, gema, theta):
    x_star = x
    search_domain = []
    for i in range(x.size()[0]):
        for j in range(x.size()[1]):
            search_domain.append([i, j])
    max_iter = int((784 * gema) / 200)
    _, source_class = torch.max(out, 0)
    print("original class {0}".format(source_class))
    print("original output {0}".format(out))
    _, target_class = torch.max(y_star, 0)
    iter = 0
    _,new_out = net(x_star.reshape(1,1,28,28))
    new_out = new_out[0]
    while source_class != target_class and iter < max_iter and search_domain != []:
        st = time.time()
        matrix = get_forward_gradient(new_out, x_star)
        ft = time.time()
        print(ft-st)
        p1, p2 = saliency_map(matrix, search_domain, target_class)
        fft = time.time()
        print(fft-ft)
        x_star[p1[0]][p1[1]] += theta
        x_star[p2[0]][p2[1]] += theta
        x_star[p1[0]][p1[1]] = judge(x_star[p1[0]][p1[1]])
        x_star[p2[0]][p2[1]] = judge(x_star[p1[0]][p1[1]])
        if x_star[p1[0]][p1[1]] == 0 or x_star[p1[0]][p1[1]] == 1:
            search_domain.remove(p1)
        if (x_star[p2[0]][p2[1]] == 0 or x_star[p2[0]][p2[1]] == 1) and p2 in search_domain:
            search_domain.remove(p2)
        x_star = x_star.detach()
        x_star.requires_grad_()
        _, new_out = net(x_star.reshape(1, 1, 28, 28))
        new_out = new_out[0]
        _,source_class = torch.max(new_out, 0)
        print("after {0} times,output is {1},prediction is {2}".format(iter,new_out,source_class))
        iter += 1
        print(time.time()-fft)

    return x_star


def get_forward_gradient(out, x):
    result = torch.zeros([out.size()[0], x.size()[0], x.size()[1]]).cuda()
    for i in range(len(out)):
        out[i].backward(retain_graph=True)
        result[i] = x.grad
        x.grad.data.zero_()
    return result


def saliency_map(matrix, search_place, target):
    p1 = [0, 0]
    p2 = [0, 0]
    max = 0
    i = 0
    for point_one in search_place:
        st = time.time()
        for point_two in search_place:
            alpha = 0
            alpha += matrix[target][point_one[0]][point_one[1]]
            alpha += matrix[target][point_two[0]][point_two[1]]
            beta = 0
            for j in range(classes):
                if j == target:
                    continue
                beta += matrix[j][point_one[0]][point_one[1]]
                beta += matrix[j][point_two[0]][point_two[1]]
            if alpha > 0 and beta < 0 and -alpha * beta > max:
                p1 = point_one
                p2 = point_two
                max = -alpha * beta
        i+=1
        print(time.time()-st)
    print(p1,p2)
    return p1, p2


net = model_define.net().cuda()
net.load_state_dict(torch.load('model.pt'))

test_set = dataset.MNIST('../../Dataset/', train=False, transform=transform.ToTensor(), download=False)
tese_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

for image, label in tese_loader:
    break
image = image.cuda()
x = image.reshape(28, 28)
x.requires_grad_()
input = x.reshape(1, 1, 28, 28)
_, output = net(input)
output = output[0]
matrix = get_forward_gradient(output, x)
search_place = []
for i in range(x.size()[0]):
    for j in range(x.size()[1]):
        search_place.append([i, j])
search_place = torch.tensor(search_place).cuda()
x_star = crafting(x, torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).cuda(), output, gema=100000, theta=1)
x_star = x_star.detach()