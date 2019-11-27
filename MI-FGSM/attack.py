import torchvision
import torch
import copy
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import pretrainedmodels

mu = 1
inception_v3 = torchvision.models.inception_v3(pretrained=True)
inception_v4 = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
incres_v2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes = 1000,pretrained = 'imagenet')
# resnet_152 = pretrainedmodels.__dict__['resnet152'](num_classes = 1000,pretrained = 'imagenet')
resnet_152 = torchvision.models.resnet152(pretrained=True)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 将image转化为tensor
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])  # 三个通道的均值和方差
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
classes = 1000
criteria = nn.CrossEntropyLoss()


# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + self.names_list[idx].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = io.imread(image_path)  # use skitimage
        label = int(self.names_list[idx].split(' ')[1])

        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def attack_one_time(white, black_list, dataset):
    criteria = nn.CrossEntropyLoss()
    total = 0
    white_mi_fgsm = 0
    white_i_fgsm = 0
    white_fgsm = 0
    black_mi_fgsm = [0 for i in range(len(black_list))]
    black_i_fgsm = [0 for i in range(len(black_list))]
    black_fgsm = [0 for i in range(len(black_list))]

    for (cnt, i) in enumerate(dataset):
        image = i['image'].unsqueeze(0)
        image = image * 2 - 1
        label = i['label']
        _, pred = torch.max(white(image), 1)
        
        if pred == label:
            total += 1

            x_attack_mi_fgsm = MI_FGSM(white, criteria, image, torch.tensor([label]).long(), 16 / 256, 10, 1)
            x_attack_i_fgsm = I_FGSM(white, criteria, image, torch.tensor([label]).long(), 16 / 256, 10)
            x_attack_fgsm = FGSM(white, criteria, image, torch.tensor([label]).long(), 16 / 256)

            _, new_pred_mi_fgsm = torch.max(white(x_attack_mi_fgsm), 1)
            _, new_pred_i_fgsm = torch.max(white(x_attack_i_fgsm), 1)
            _, new_pred_fgsm = torch.max(white(x_attack_fgsm), 1)

            if (label != new_pred_mi_fgsm.item()):
                white_mi_fgsm += 1
            if (label != new_pred_i_fgsm.item()):
                white_i_fgsm += 1
            if (label != new_pred_fgsm.item()):
                white_fgsm += 1

            for i in range(len(black_list)):
                _, pred_mi_fgsm = torch.max(black_list[i](x_attack_mi_fgsm), 1)
                if (label != pred_mi_fgsm.item()):
                    black_mi_fgsm[i] += 1

            for i in range(len(black_list)):
                _, pred_i_fgsm = torch.max(black_list[i](x_attack_i_fgsm), 1)
                if (label != pred_i_fgsm.item()):
                    black_i_fgsm[i] += 1

            for i in range(len(black_list)):
                _, pred_fgsm = torch.max(black_list[i](x_attack_fgsm), 1)
                if (label != pred_fgsm.item()):
                    black_fgsm[i] += 1

        if total == 10:
            break
    print("MI_FGSM attack rate : ", white_mi_fgsm / total)
    print("I_FGSM attack rate : ", white_i_fgsm / total)
    print("FGSM attack rate : ", white_fgsm / total)
    print("balck_mi_fgsm attack : ", black_mi_fgsm / total)
    print("balck_i_fgsm attack : ", black_i_fgsm / total)
    print("balck_fgsm attack : ", black_fgsm / total)


def ensemble_pred(ensemble_list, image, weight_list):
    output = ensemble_list[0](image) * weight_list[0]
    for i in range(1, len(ensemble_list)):
        output += ensemble_list[i](image) * weight_list[i]
    _, pred = torch.max(output, 1)
    return pred


def attack_one_time_ensemble(dataset, ensemble_model_list, weight_list, exclude_model):
    total = 0
    white_mi_fgsm = [0,0,0]
    white_i_fgsm = [0,0,0]
    white_fgsm = [0,0,0]
    black_mi_fgsm = [0,0,0]
    black_i_fgsm = [0,0,0]
    black_fgsm = [0,0,0]
    resize = transforms.Resize(224)
    toimage = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    for (cnt, i) in enumerate(dataset):
        image = i['image'].unsqueeze(0)
        image = image * 2 - 1
        label = i['label']
        pred = ensemble_pred(ensemble_model_list,image,weight_list)
        print(label)
        if pred == label:
            total += 1


            x_attack_mi_fgsm_logits = MI_FGSM_ENSEMBLE('logits',ensemble_model_list,weight_list,image,torch.tensor([label]),32/255,10,1)
            x_attack_i_fgsm_logits = I_FGSM_ENSEMBLE('logits',ensemble_model_list,weight_list,image,torch.tensor([label]),32/255,10)
            x_attack_fgsm_logits = FGSM_ENSEMBLE('logits',ensemble_model_list,weight_list,image,torch.tensor([label]),32/255)

            x_attack_mi_fgsm_predctions = MI_FGSM_ENSEMBLE('predictions', ensemble_model_list, weight_list, image, torch.tensor([label]),
                                                       32 / 255, 10, 1)
            x_attack_i_fgsm_predctions = I_FGSM_ENSEMBLE('predictions', ensemble_model_list, weight_list, image, torch.tensor([label]), 32 / 255,
                                                     10)
            x_attack_fgsm_predctions = FGSM_ENSEMBLE('predictions', ensemble_model_list, weight_list, image, torch.tensor([label]), 32 / 255)

            x_attack_mi_fgsm_loss = MI_FGSM_ENSEMBLE('loss', ensemble_model_list, weight_list, image,
                                                     torch.tensor([label]),
                                                     32 / 255, 10, 1)
            x_attack_i_fgsm_loss = I_FGSM_ENSEMBLE('loss', ensemble_model_list, weight_list, image,
                                                   torch.tensor([label]), 32 / 255,
                                                   10)
            x_attack_fgsm_loss = FGSM_ENSEMBLE('loss', ensemble_model_list, weight_list, image, torch.tensor([label]),
                                               32 / 255)

            pred_mi_fgsm_logits = ensemble_pred(ensemble_model_list,x_attack_mi_fgsm_logits,weight_list)
            pred_i_fgsm_logits = ensemble_pred(ensemble_model_list,x_attack_i_fgsm_logits,weight_list)
            pred_fgsm_logits = ensemble_pred(ensemble_model_list,x_attack_fgsm_logits,weight_list)

            pred_mi_fgsm_predctions = ensemble_pred(ensemble_model_list,x_attack_mi_fgsm_predctions,weight_list)
            pred_i_fgsm_predctions = ensemble_pred(ensemble_model_list,x_attack_i_fgsm_predctions,weight_list)
            pred_fgsm_predctions = ensemble_pred(ensemble_model_list,x_attack_fgsm_predctions,weight_list)

            pred_mi_fgsm_loss = ensemble_pred(ensemble_model_list,x_attack_mi_fgsm_loss,weight_list)
            pred_i_fgsm_loss = ensemble_pred(ensemble_model_list,x_attack_i_fgsm_loss,weight_list)
            pred_fgsm_loss = ensemble_pred(ensemble_model_list,x_attack_fgsm_loss,weight_list)

            update(white_mi_fgsm,[pred_mi_fgsm_logits,pred_mi_fgsm_predctions,pred_mi_fgsm_loss],label)
            update(white_i_fgsm,[pred_i_fgsm_logits,pred_i_fgsm_predctions,pred_i_fgsm_loss],label)
            update(white_fgsm,[pred_fgsm_logits,pred_fgsm_predctions,pred_fgsm_loss],label)


            _,b_mi_fgsm_logits = torch.max(exclude_model(x_attack_mi_fgsm_logits),1)
            _,b_i_fgsm_logits = torch.max(exclude_model(x_attack_i_fgsm_logits),1)
            _,b_fgsm_logits = torch.max(exclude_model(x_attack_fgsm_logits),1)
            _,b_mi_fgsm_predctions = torch.max(exclude_model(x_attack_mi_fgsm_predctions),1)
            _,b_i_fgsm_predctions = torch.max(exclude_model(x_attack_i_fgsm_predctions),1)
            _,b_fgsm_predctions = torch.max(exclude_model(x_attack_fgsm_predctions),1)
            _,b_mi_fgsm_loss = torch.max(exclude_model(x_attack_mi_fgsm_loss),1)
            _,b_i_fgsm_loss = torch.max(exclude_model(x_attack_i_fgsm_loss),1)
            _,b_fgsm_loss = torch.max(exclude_model(x_attack_fgsm_loss),1)

            update(black_mi_fgsm,[b_mi_fgsm_logits,b_mi_fgsm_predctions,b_mi_fgsm_loss],label)
            update(black_i_fgsm,[b_i_fgsm_logits,b_i_fgsm_predctions,b_i_fgsm_loss],label)
            update(black_fgsm,[b_fgsm_logits,b_fgsm_predctions,b_fgsm_loss],label)

        if total == 1:
            break
    print("whits mi_fgsm : " ,[white_mi_fgsm[i]/total for i in range(len(white_mi_fgsm))])
    print("whits i_fgsm : " ,[white_i_fgsm[i]/total for i in range(len(white_i_fgsm))])
    print("whits fgsm : " ,[white_fgsm[i]/total for i in range(len(white_fgsm))])
    print("black mi_fgsm : " ,[black_mi_fgsm[i]/total for i in range(len(black_mi_fgsm))])
    print("black i_fgsm : " ,[black_i_fgsm[i]/total for i in range(len(black_i_fgsm))])
    print("black fgsm : " ,[black_fgsm[i]/total for i in range(len(black_fgsm))])

def update(lis,pre_lis,label):
    for i in range(len(pre_lis)):
        if pre_lis[i] != label:
            lis[i] += 1
def white_box_attack(model, image, label):
    _, pred = torch.max(model(image), 1)

    if pred == label:

        x_attack_mi_fgsm = MI_FGSM(model, criteria, image, torch.tensor([label]).long(), 16 / 256, 10, 1)
        x_attack_i_fgsm = I_FGSM(model, criteria, image, torch.tensor([label]).long(), 16 / 256, 10)
        x_attack_fgsm = FGSM(model, criteria, image, torch.tensor([label]).long(), 16 / 256)

        _, new_pred_mi_fgsm = torch.max(model(x_attack_mi_fgsm), 1)
        _, new_pred_i_fgsm = torch.max(model(x_attack_i_fgsm), 1)
        _, new_pred_fgsm = torch.max(model(x_attack_fgsm), 1)
        return x_attack_mi_fgsm, x_attack_i_fgsm, x_attack_fgsm, new_pred_mi_fgsm, new_pred_i_fgsm, new_pred_fgsm
    else:
        return None


def black_box_attack(model, mi_fgsm, i_fgsm, fgsm):
    _, pred_mi_fgsm = torch.max(model(mi_fgsm), 1)
    _, pred_i_fgsm = torch.max(model(i_fgsm), 1)
    _, pred_fgsm = torch.max(model(fgsm), 1)
    return pred_mi_fgsm, pred_i_fgsm, pred_fgsm


def show_image(image):
    plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))


def MI_FGSM(model, criteria, original_input, true_label, pertubation, iterations, decay):
    alpha = pertubation / iterations
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()
    g = torch.zeros(x_star.shape)
    for t in range(iterations):
        output = model(x_star)
        loss = criteria(output, true_label)
        loss.backward()
        # print(x_star.grad)
        gradient = x_star.grad

        g = decay * g + gradient / np.linalg.norm(
            gradient.reshape([gradient.shape[0] * gradient.shape[1] * gradient.shape[2] * gradient.shape[3]]), ord=1)
        x_star = x_star + alpha * torch.sign(g)
        x_star.detach_()
        x_star.requires_grad_()
    return x_star


def MI_FGSM_ENSEMBLE(ensemble_way, model_list, weight_list, original_input, true_label, pertubation, iterations, decay):
    alpha = pertubation / iterations
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()
    g = torch.zeros(x_star.shape)
    for t in range(iterations):

        if ensemble_way == 'logits':
            output = weight_list[0] * model_list[0](x_star)
            for i in range(1, len(model_list)):
                output += weight_list[i] * model_list[i](x_star)
            lsm = nn.LogSoftmax()
            output = lsm(output)
            loss = -torch.sum(torch.matmul(torch.eye(1000)[true_label], output[0]))
        elif ensemble_way == 'predictions':
            sm = nn.Softmax()
            output = weight_list[0] * sm(model_list[0](x_star))
            for i in range(1, len(model_list)):
                output += weight_list[i] * sm(model_list[i](x_star))
            output = torch.log(output)
            loss = -torch.sum(torch.matmul(torch.eye(1000)[true_label], output[0]))

        else:
            cl = nn.CrossEntropyLoss()
            output = weight_list[0] * cl(model_list[0](x_star), true_label.long())
            for i in range(1, len(model_list)):
                output += weight_list[i] * cl(model_list[i](x_star), true_label.long())
            loss = output

        loss.backward()
        # print(x_star.grad)
        gradient = x_star.grad

        g = decay * g + gradient / np.linalg.norm(
            gradient.reshape([gradient.shape[0] * gradient.shape[1] * gradient.shape[2] * gradient.shape[3]]), ord=1)
        x_star = x_star + alpha * torch.sign(g)
        x_star.detach_()
        x_star.requires_grad_()
    return x_star


def I_FGSM(model, criteria, original_input, true_label, pertubation, iterations):
    alpha = pertubation / iterations
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()
    for t in range(iterations):
        output = model(x_star)
        loss = criteria(output, true_label)
        loss.backward()
        # print(x_star.grad)
        g = x_star.grad
        x_star = x_star + alpha * torch.sign(g).cuda()
        x_star.detach_()
        x_star.requires_grad_()
    return x_star


def I_FGSM_ENSEMBLE(ensemble_way, model_list, weight_list, original_input, true_label, pertubation, iterations):
    alpha = pertubation / iterations
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()
    for t in range(iterations):
        if ensemble_way == 'logits':
            output = weight_list[0] * model_list[0](x_star)
            for i in range(1, len(model_list)):
                output += weight_list[i] * model_list[i](x_star)
            lsm = nn.LogSoftmax()
            output = lsm(output)
            loss = -torch.sum(torch.matmul(torch.eye(1000)[true_label], output[0]))
        elif ensemble_way == 'predictions':
            sm = nn.Softmax()
            output = weight_list[0] * sm(model_list[0](x_star))
            for i in range(1, len(model_list)):
                output += weight_list[i] * sm(model_list[i](x_star))
            output = torch.log(output)
            loss = -torch.sum(torch.matmul(torch.eye(1000)[true_label], output[0]))

        else:
            cl = nn.CrossEntropyLoss()
            output = weight_list[0] * cl(model_list[0](x_star), true_label.long())
            for i in range(1, len(model_list)):
                output += weight_list[i] * cl(model_list[i](x_star), true_label.long())
            loss = output

        loss.backward()
        # print(x_star.grad)
        g = x_star.grad

        x_star = x_star + alpha * torch.sign(g)
        x_star.detach_()
        x_star.requires_grad_()
    return x_star


def FGSM(model, criteria, original_input, true_label, pertubation):
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()
    output = model(x_star)
    loss = criteria(output, true_label)
    loss.backward()
    # print(x_star.grad)
    g = x_star.grad
    x_star = x_star + pertubation * torch.sign(g).cuda()
    return x_star


def FGSM_ENSEMBLE(ensemble_way, model_list, weight_list, original_input, true_label, pertubation):
    x_star = copy.deepcopy(original_input)
    x_star.requires_grad_()

    if ensemble_way == 'logits':
        output = weight_list[0] * model_list[0](x_star)
        for i in range(1, len(model_list)):
            output += weight_list[i] * model_list[i](x_star)
        lsm = nn.LogSoftmax()
        output = lsm(output)
        loss = -torch.sum(torch.matmul(torch.eye(1000)[true_label], output[0]))
    elif ensemble_way == 'predictions':
        sm = nn.Softmax()
        output = weight_list[0] * sm(model_list[0](x_star))
        for i in range(1, len(model_list)):
            output += weight_list[i] * sm(model_list[i](x_star))
        output = torch.log(output)
        loss = -torch.sum(torch.matmul(torch.eye(1000)[true_label], output[0]))

    else:
        cl = nn.CrossEntropyLoss()
        output = weight_list[0] * cl(model_list[0](x_star), true_label.long())
        for i in range(1, len(model_list)):
            output += weight_list[i] * cl(model_list[i](x_star), true_label.long())
        loss = output

    loss.backward()
    # print(x_star.grad)
    g = x_star.grad

    x_star = x_star + pertubation * torch.sign(g)
    x_star.detach_()
    x_star.requires_grad_()
    return x_star


if __name__ == '__main__':
    testset = MyDataset('../dataset/dataset/images/', '../dataset/dataset/labels',transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
    success_mi_fgsm = 0
    success_i_fgsm = 0
    success_fgsm = 0
    total = 0
    inception_v3.eval()
    inception_v4.eval()
    incres_v2.eval()
    resnet_152.eval()
    image = attack_one_time_ensemble(testset,[inception_v3,inception_v4,incres_v2],[1/3,1/3,1/3],inception_v3)

