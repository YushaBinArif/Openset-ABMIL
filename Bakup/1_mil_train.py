# -*- coding: utf-8 -*-
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import scipy.spatial.distance as spd
import libmr
import random
from torchvision.transforms import transforms

#sys.path.append('..')
#sys.path.append(os.pardir)
import util_openmax #import BagMNIST, TestMNIST
from dataset import bagDataset, insDataset

# Import necessary libraries
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#import seaborn as sns

# set seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_class_num = 3

class_n = 0
class_p1 = 1
class_p2 = 2
class_u = 3

number_of_bags = 1000
bag_size = 10

#openmax setting
weibull_tail = 20
weibull_alpha = 2
weibull_threshold = 0.6

#正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    if(y_hat != true_label):
        ans = 0
    return ans

# define model
class MIL(nn.Module):
    def __init__(self, n_class):
        super(MIL, self).__init__()

        self.n_class = n_class

        self.feature_ex = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

        self.attention = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_class)
        )

    def forward(self, input):
        x = input.squeeze(0)  # (num_instance x 784)
        h = self.feature_ex(x) # (num_instance x 16)

        a = self.attention(h)           # (num_instance x 1)
        a_t = torch.transpose(a, 1, 0)  # (1 x num_instance)
        a_n = F.softmax(a_t, dim=1)     # (1 x num_instance)

        z = torch.mm(a_n, h) # (1 x 16)
        class_prob = self.classifier(z).reshape(1, self.n_class) # (1 x n_class)

        return class_prob, a_n

class insMLP(nn.Module):
    def __init__(self, n_class):
        super(insMLP, self).__init__()

        self.n_class = n_class

        self.feature_ex = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_class)
        )

    def forward(self, x):

        x = input.squeeze(0)  # (num_instance x 784)
        h = self.feature_ex(x) # (num_instance x 16)
        class_prob = self.classifier(h).reshape(-1, self.n_class) # (1 x n_class)

        return class_prob, h

def train(model, optimizer, train_loader):

    # model = model.to(device)
    model.train() #訓練モードに変更
    train_class_loss = 0.0
    correct_num = 0
    bag_all = 0

    for (input_tensor, class_label) in train_loader:
        # input_tensor = input_tensor.to(device)
        # class_label = class_label.to(device)

        optimizer.zero_grad() # initialization of gradient
        class_prob, attention = model(input_tensor[0]) # class probability
        # print(f"class_prob  {class_prob} \n class_label  {class_label[0]}")
        # print(f"attention  {attention}")


        class_loss = F.cross_entropy(class_prob, class_label[0])
        train_class_loss += class_loss.item()
        class_loss.backward() # backpropagation
        optimizer.step() # update parameters

        class_hat = torch.argmax(class_prob)
        correct_num += eval_ans(class_hat, class_label[0])
        bag_all += 1

    return train_class_loss/bag_all, correct_num/bag_all




def validate(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (input_tensor, class_label) in val_loader:
            class_prob, attention = model(input_tensor[0])
            class_loss = F.cross_entropy(class_prob, class_label[0])
            val_loss += class_loss.item()

            class_hat = torch.argmax(class_prob)
            if class_hat.item() == class_label[0].item():
                correct += 1
            total += 1

    avg_loss = val_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

####################################################################
####################################################################
####################################################################
# main function
####################################################################
####################################################################    
####################################################################


model = MIL(n_class=3)
model.to(device)

EPOCHS = 10

import pandas as pd

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# model読み込み
# from model import MIL
# 各ブロック宣言
model = MIL(n_class=3)

optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# pre-processing
#transform = torchvision.transforms.Compose([
#    torchvision.transforms.ToTensor()
#])

train_loss = 0.0
train_acc = 0.0

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        mnist_train = bagDataset(
            split='train',
            num_bag=2000,
            bag_size=10,
            epoch=epoch
        )
        mnist_val = bagDataset(
            split='val',
            num_bag=500,
            bag_size=10,
            epoch=epoch
        )

        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            mnist_val,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )

        class_loss, acc = train(model, optimizer, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        scheduler.step()

        print(f"Epoch {epoch}: Train Loss={class_loss:.4f}, Train Acc={acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")


"""## Save the model as ONNX"""

# save pth
model.eval()
pth_path = f'./models/mnist_openmil.pth'
torch.save(model, pth_path)
