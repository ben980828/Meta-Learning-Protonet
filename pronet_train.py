import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Function
from sklearn import preprocessing
import imageio
import scipy.misc
import argparse
import glob
import os
import sys
import csv
import pandas as pd
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


class MiniImageNet(Dataset):
    def __init__(self, fileroot, image_root, label_csv, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.fileroot = fileroot
        self.image_root = image_root
        self.transform = transform
        self.label_csv = label_csv
        self.y = self.label_csv[1:,2]
        self.len = len(self.image_root) 

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.image_root[index]
        image_path = os.path.join(self.fileroot, image_fn)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image) 

        label = self.label_csv[np.where(self.label_csv==image_fn)[0].item()][2]
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProNet(nn.Module):

    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super(ProNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, input_data):
        x = self.encoder(input_data)
        return x.view(x.size(0), -1)

class ProBatchSampler(object):
    def __init__(self, labels, classes_it, shots, episodes):
        self.episodes = episodes
        self.classes_it = classes_it
        self.shots = shots
        self.le = preprocessing.LabelEncoder()
        labels = self.le.fit_transform(labels)# encode all classes into int
        labels = np.array(labels)
        self.class_index = []
        for i in range(max(labels) + 1):
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.class_index.append(index)

    def __len__(self):
        return self.episodes
    
    def __iter__(self):
        for i_batch in range(self.episodes):
            batch = []
            classes = torch.randperm(len(self.class_index))[:self.classes_it]
            for c in classes:
                l = self.class_index[c]
                pos = torch.randperm(len(l))[:self.shots]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class Param_Dist(nn.Module):
    def __init__(self, n, m):
        super(Param_Dist, self).__init__()
        self.N = n
        self.M = m
        self.W = torch.nn.Parameter(torch.rand(n, m))
    def forward(self, x, y):
        n = x.shape[0]
        m = y.shape[0]
        a = x.unsqueeze(1).expand(n, m, -1)
        b = y.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)*self.W).sum(dim=2)

        return logits

def euclidean_dist(x, y):
    n = x.shape[0]
    m = y.shape[0]
    a = x.unsqueeze(1).expand(n, m, -1)
    b = y.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)

    return logits

def cosine_sim(x, y):
    n = x.shape[0]
    m = y.shape[0]
    a = x.unsqueeze(1).expand(n, m, -1)
    b = y.unsqueeze(0).expand(n, m, -1)
    cos = nn.CosineSimilarity(dim=2)
    logits = cos(a, b)

    return logits

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def main():
    classes_per_it = 5
    val_classes_per_it = 5
    n_support = 1
    n_query = 1
    episodes = 600
    mode = 'euc'



    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    train_root = './hw4-ben980828/hw4_data/train/'
    train_label_path = './hw4-ben980828/hw4_data/train.csv'

    train_list = os.listdir(train_root)

    train_label_file = pd.read_csv(train_label_path, sep=',',header=None)
    train_label_matrix = train_label_file.to_numpy()


    val_root = './hw4-ben980828/hw4_data/val/'
    val_label_path = './hw4-ben980828/hw4_data/val.csv'

    val_list = os.listdir(val_root)

    val_label_file = pd.read_csv(val_label_path, sep=',',header=None)
    val_label_matrix = val_label_file.to_numpy()
        
    train_transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
    val_transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])


    train_set = MiniImageNet(fileroot=train_root, 
        image_root=train_list,
        label_csv=train_label_matrix,  
        transform=train_transform
        )
    val_set = MiniImageNet(fileroot=val_root, 
        image_root=val_list, 
        label_csv=val_label_matrix, 
        transform=val_transform
        )
        
    train_sampler = ProBatchSampler(labels=train_set.y, classes_it=classes_per_it,
                                    shots=n_support+n_query, episodes=episodes)
    val_sampler = ProBatchSampler(labels=val_set.y, classes_it=val_classes_per_it,
                                    shots=n_support+n_query, episodes=episodes)

    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=1)
    valid_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=1)


    pronet = ProNet()
    pronet = pronet.cuda()

    print(pronet)
    

    lr = 1e-3
    loss_proto = nn.CrossEntropyLoss()
    if mode == 'euc' or mode == 'cos':
        optimizer = optim.Adam(pronet.parameters(), lr=lr, betas=(0.5, 0.999))
    elif mode == 'param':
        param_dist = Param_Dist().cuda()
        param = list(pronet.parameters())
        param.extend(list(param_dist.parameters()))
        optimizer = optim.Adam(param, lr=lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)


    epoch = 60
    iteration = 0
    ep_list = []
    val_iteration = 0
    max_acc = 0.
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    iter_list = []
    val_iter_list = []
    # training
    for ep in range(1, epoch+1):
        pronet.train()
        print('Current training epoch : ', ep)
        for i, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            k = classes_per_it * n_support
            data_sup, data_que = data[:k], data[k:]
            data_sup, data_que = data_sup.cuda(), data_que.cuda()

            model_output = pronet(data_sup)
            output = model_output.reshape(n_support, classes_per_it, -1).mean(dim=0)# means of classes along n_support shots

            model_label = torch.arange(classes_per_it).repeat(n_query)# label for classify diff classes
            label = model_label.type(torch.cuda.LongTensor)

            if mode == 'euc':
                logits = euclidean_dist(pronet(data_que), output)
            elif mode == 'cos':
                logits = cosine_sim(pronet(data_que), output)
            elif mode == 'param':
                logits = param_dist(pronet(data_que), output)

            loss = loss_proto(logits, label)
            acc = count_acc(logits, label)

            loss.backward()
            optimizer.step()
            
            iteration += 1
            if iteration%50 == 0:
                iter_list.append(iteration)
                train_loss_list.append(loss.item())
                train_acc_list.append(acc)
            sys.stdout.write('\r epoch: %d, [iter: %d / all: %d], loss: %f, acc: %f' \
                    % (ep, i + 1, len(train_loader), loss.data.cpu().numpy(), acc
                )
            )
            sys.stdout.flush()

        pronet.eval()

        with torch.no_grad(): # This will free the GPU memory used for back-prop
            total_acc = []
            mean_acc = 0.
            std_acc = 0.
            report_acc_up = 0.
            report_acc_low = 0.
            for i, (data, _) in enumerate(valid_loader):
                k = val_classes_per_it * n_support
                data_sup, data_que = data[:k], data[k:]
                data_sup, data_que = data_sup.cuda(), data_que.cuda()

                model_output = pronet(data_sup)
                output = model_output.reshape(n_support, val_classes_per_it, -1).mean(dim=0)# means of classes along n_support shots

                model_label = torch.arange(val_classes_per_it).repeat(n_query)# label for classify diff classes
                label = model_label.type(torch.cuda.LongTensor)

                if mode == 'euc':
                    logits = euclidean_dist(pronet(data_que), output)
                elif mode == 'cos':
                    logits = cosine_sim(pronet(data_que), output)
                elif mode == 'param':
                    logits = param_dist(pronet(data_que), output)
                    
                loss = loss_proto(logits, label)
                acc = count_acc(logits, label)
                total_acc.append(acc)

                val_iteration += 1
                if val_iteration%50 == 0:
                    val_iter_list.append(val_iteration)
                    val_loss_list.append(loss.item())
                # sys.stdout.write('\r epoch: %d, [iter: %d / all: %d], loss: %f, acc: %f' \
                #         % (ep, i + 1, len(valid_loader), loss.data.cpu().numpy(), acc
                #     )
                # )
                # sys.stdout.flush()
            mean_acc = sum(total_acc) / len(valid_loader)
            std_acc = np.array(total_acc).std().item()
            report_acc_up = mean_acc + 1.96*(std_acc/600**0.5)
            report_acc_low = mean_acc - 1.96*(std_acc/600**0.5)
            val_acc_list.append(mean_acc)
            print('\nValidation Set Acc: \nupper bound: {}, lower bound: {}'.format(report_acc_up, report_acc_low))

            if report_acc_up > max_acc:
                print('Performance improved : ({:.3f} --> {:.3f}). Save model ==> '.format(max_acc, report_acc_up))
                max_acc = report_acc_up
                torch.save(pronet.state_dict(), 'pronet_{}shots_{}_ep{}_acc{}±{}.pth'.format(n_support, mode, ep, mean_acc, 1.96*(std_acc/600**0.5)))
        lr_scheduler.step()
        ep_list.append(ep)

        #lr_decay.step(mean_loss)



    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0][0].plot(iter_list, train_loss_list)
    ax[0][0].set_title('Train Loss')
    ax[0][0].set(xlabel="iteration", ylabel="Loss Value")

    ax[0][1].plot(iter_list, train_acc_list)
    ax[0][1].set_title('Train Acc')
    ax[0][1].set(xlabel="iteration", ylabel="acc")

    ax[1][0].plot(iter_list, val_loss_list)
    ax[1][0].set_title('Val Loss')
    ax[1][0].set(xlabel="iteration", ylabel="Loss Value")

    ax[1][1].plot(ep_list, val_acc_list)
    ax[1][1].set_title('Val Acc')
    ax[1][1].set(xlabel="epoch", ylabel="acc")

    plt.savefig('Loss_Curve_ProtoNet.png')

if __name__ == '__main__':
    main()
