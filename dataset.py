# In this version I am dividing the negative instances into two halves
# and using one half for positive1 and the other half for positive2.    

import numpy as np
import torch
from torch.utils.data import Dataset
#from torchvision.transforms import transforms
import pandas as pd
import random

class_n = 0
class_p1 = 1
class_p2 = 2
class_u = 3

class bagDataset(torch.utils.data.Dataset): # dataloader with bag labels
    def __init__(self, num_bag=1000, bag_size=10, split = 'train', transform = None, epoch=0):

        self.bags = []
        self.labels = []
        self.split = split
        self.transform = transform
        self.bag_size = bag_size
        self.num_bag = num_bag

        ins_dim = 784

        if split == 'train':
            df = pd.read_csv('./data/mnist_split_train.csv', header=None)
        elif split == 'val':
            df = pd.read_csv('./data/mnist_split_val.csv', header=None)
        elif split == 'test':
            df = pd.read_csv('./data/mnist_split_test.csv', header=None)

        array = df.to_numpy()
        label = array[:,-1]
        data = (array[:,:-1]/255.0).astype(np.float32)

        n_idx = (label == class_n)
        n_data_all = data[n_idx]
        np.random.shuffle(n_data_all)  # ensure randomness

        # Split into two halves
        half = len(n_data_all) // 2
        n_data_for_neg = n_data_all[:half]
        n_data_for_pos = n_data_all[half:]
        

        p1_idx = (label == class_p1)
        p1_data = data[p1_idx]
        p2_idx = (label == class_p2)
        p2_data = data[p2_idx]

        if split == 'train':
            random.seed(epoch)
        else:
            random.seed(0)

        for _ in range(self.num_bag):  # create negative bags
            np.random.shuffle(n_data_for_neg)
            instances = n_data_for_neg[0].reshape(ins_dim)
            for i in range(self.bag_size - 1):
                tmp = n_data_for_neg[i + 1].reshape(ins_dim)
                instances = np.vstack([instances, tmp])
            self.bags.append(torch.from_numpy(instances).float())
            self.labels.append(torch.from_numpy(np.array([0])).long())

        for _ in range(self.num_bag): # create positive1 bags
            num_pins = np.random.randint(1, self.bag_size + 1)
            np.random.shuffle(p1_data)
            np.random.shuffle(n_data_for_pos)
            instances = p1_data[0].reshape(ins_dim)
            # stack positive instances
            for i in range(num_pins - 1):
                tmp = p1_data[i+1].reshape(ins_dim)
                instances = np.vstack([instances, tmp])
            # stack negative instances
            for i in range(self.bag_size - num_pins):
                tmp = n_data_for_pos[i].reshape(ins_dim)
                instances = np.vstack([instances, tmp])
            self.bags.append(torch.from_numpy(instances).float())
            self.labels.append(torch.from_numpy(np.array([1])).long())

        for _ in range(self.num_bag): # create positive2 bags
            num_pins = np.random.randint(1, self.bag_size + 1)
            np.random.shuffle(p2_data)
            np.random.shuffle(n_data_for_pos)
            instances = p2_data[0].reshape(ins_dim)
            # stack positive instances
            for i in range(num_pins - 1):
                tmp = p2_data[i+1].reshape(ins_dim)
                instances = np.vstack([instances, tmp])
            # stack negative instances
            for i in range(self.bag_size - num_pins):
                tmp = n_data_for_pos[i].reshape(ins_dim)
                instances = np.vstack([instances, tmp])
            self.bags.append(torch.from_numpy(instances).float())
            self.labels.append(torch.from_numpy(np.array([2])).long())

        self.data_num = len(self.bags)

    def __len__(self):

        return self.data_num

    def __getitem__(self, idx):

        return self.bags[idx], self.labels[idx]

class insDataset(torch.utils.data.Dataset): # dataloader with instance labels
    def __init__(self, train = True, transform = None, epoch=0):

        self.train = train
        self.transform = transform

        ins_dim = 784

        if train:
            df = pd.read_csv(f'mnist_train.csv',header=None)
        else:
            df = pd.read_csv(f'mnist_test.csv',header=None)
        array = df.to_numpy()
        label = array[:,-1]
        data = (array[:,:-1]/255.0).astype(np.float32)

        n_idx = (label == class_n)
        n_data = data[n_idx]
        n_label = label[n_idx]
        p1_idx = (label == class_p1)
        p1_data = data[p1_idx]
        p1_label = label[p1_idx]
        p2_idx = (label == class_p2)
        p2_data = data[p2_idx]
        p2_label = label[p2_idx]
        if not train:
            u_idx = (label == class_u)
            u_data = data[u_idx]
            u_label = label[u_idx]

        self.instances = n_data
        self.instances = np.vstack([self.instances, p1_data])
        self.instances = np.vstack([self.instances, p2_data])

        self.labels = n_label
        self.labels = np.concatenate([self.labels, p1_label])
        self.labels = np.concatenate([self.labels, p2_label])
#        self.labels = np.vstack([self.labels, p1_label])
#        self.labels = np.vstack([self.labels, p2_label])

        if not train:
            self.instances = np.vstack([self.instances, u_data])
            self.labels = np.concatenate([self.labels, u_label])

        self.instances = torch.from_numpy(self.instances).float()
        self.labels = torch.from_numpy(self.labels).long()

        self.data_num = self.instances.shape[0]

    def __len__(self):

        return self.data_num

    def __getitem__(self, idx):

        return self.instances[idx], self.labels[idx]
