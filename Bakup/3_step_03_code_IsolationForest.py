#The result after applying OpenMax to the instance-level features and bag-level features
#The result after improving the MAVs calculation method

#Imports
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
from libmr import MR
from collections import defaultdict
import random
from torchvision.transforms import transforms

#sys.path.append('..')
#sys.path.append(os.pardir)
import util_openmax #import BagMNIST, TestMNIST
from dataset import bagDataset, insDataset
from dataset_unk_bags import bagDataset as unk_bagDataset

# Import necessary libraries
from sklearn.metrics import confusion_matrix


#Set seed
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

#Openmax setting
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
    def __init__(self, feature_ex, classifier, n_class):
        super(insMLP, self).__init__()
        self.n_class = n_class
        self.feature_ex = feature_ex
        self.classifier = classifier


    def forward(self, input):

        x = input.squeeze(0)  # (num_instance x 784)
        h = self.feature_ex(x) # (num_instance x 16)
        class_prob = self.classifier(h).reshape(-1, self.n_class) # (1 x n_class)
        return class_prob, h

def collect_instance_embeddings(model, data_loader):
    model.eval()
    features = []
    preds = []

    with torch.no_grad():
        for images, true_labels in data_loader:
            images = images.to(device)  # shape: (batch_size, 784)
            true_labels = true_labels.to(device)  # shape: (batch_size,)

            instance_logits, _ = model(images)  # shape: (batch_size, n_class)
            predicted_labels = torch.argmax(instance_logits, dim=1)

            # Only keep correct predictions
            correct_mask = (predicted_labels == true_labels)
            correct_feats = instance_logits[correct_mask]
            correct_labels = true_labels[correct_mask]

            features.append(correct_feats.cpu())
            preds.append(correct_labels.cpu())

    all_features = torch.cat(features, dim=0).numpy()
    all_preds = torch.cat(preds, dim=0).numpy()

    return all_features, all_preds



def collect_bag_embeddings(model, data_loader):
    model.eval()
    class_feats = defaultdict(list)

    with torch.no_grad():
        for input_tensors, label_batch in data_loader:
            for i in range(input_tensors.size(0)):
                bag_tensor = input_tensors[i].unsqueeze(0)  # (1, bag_size, 784)
                true_label = label_batch[i].item()

                bag_logits, _ = model(bag_tensor)  # (1, n_class)
                pred_label = torch.argmax(bag_logits, dim=1).item()
                bag_logits_np = bag_logits.cpu().squeeze().numpy()

                if true_label == class_n:
                    # Rule 1: All negative bags go into class_n
                    class_feats[class_n].append(bag_logits_np)

                elif true_label in [class_p1, class_p2]:
                    if pred_label == true_label:
                        # Rule 2a: Correct positive prediction → add to its own MAV
                        class_feats[true_label].append(bag_logits_np)
                    elif pred_label == class_n:
                        # Rule 2b: Misclassified as negative → add to negative MAV
                        class_feats[class_n].append(bag_logits_np)
                    else:
                        # Rule 2c: Misclassified as wrong positive → ignore
                        continue
    
    # Convert dict to arrays
    all_class_means = {}
    all_weibulls = {}
    for cls in class_feats:
        feats = np.stack(class_feats[cls])
        mean_vec = np.mean(feats, axis=0)
        all_class_means[cls] = mean_vec

        # Fit Weibull
        dists = np.linalg.norm(feats - mean_vec, axis=1)
        tail = np.sort(dists)[-weibull_tail:]
        mr = MR()
        mr.fit_high(tail, len(tail))
        all_weibulls[cls] = mr

    return all_class_means, all_weibulls




def fit_openmax(features, labels, tail_size=20):
    class_feats = defaultdict(list)

    for feat, label in zip(features, labels):
        class_feats[label].append(feat)

    class_means = {}
    weibull_models = {}

    for cls in range(train_class_num):
        feats = np.stack(class_feats[cls])
        mean_vec = np.mean(feats, axis=0)
        class_means[cls] = mean_vec

        dists = np.linalg.norm(feats - mean_vec, axis=1)
        sorted_dists = np.sort(dists)[-tail_size:]
        mr = MR()
        mr.fit_high(sorted_dists, len(sorted_dists))
        weibull_models[cls] = mr

    return class_means, weibull_models

def apply_openmax(instance_feat, class_means, weibull_models, threshold=0.5):
    openmax_scores = []
    unknown_scores = []

    for cls in range(train_class_num):
        dist = np.linalg.norm(instance_feat - class_means[cls])
        w_score = weibull_models[cls].w_score(dist)
        openmax_scores.append(1 - w_score)  # known class confidence
        unknown_scores.append(w_score)

    total = sum(openmax_scores) + sum(unknown_scores)
    probs = [s / total for s in openmax_scores]
    p_unknown = sum(unknown_scores) / total

    if p_unknown > threshold:
        return 3 ,p_unknown
    else:
        pred_cls = np.argmax(probs)
        return pred_cls, p_unknown

from sklearn.svm import OneClassSVM

# Train One-Class SVM outside the function using negative class logits
def train_ocsvm(features, labels):
    neg_logits = features[labels == 0]  # assuming these are 3-dim logits
    ocsvm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.01)
    ocsvm.fit(neg_logits)
    return ocsvm

# Use One-Class SVM instead of OpenMax in attention assignment
def predict_bag_with_isolation_attention(bag_tensor, ins_model, iso_forest_model, bag_classifier):
    ins_model.eval()
    with torch.no_grad():
        x = bag_tensor.squeeze(0)  # (num_instances x 784)

        # 1. Extract features and logits
        features = ins_model.feature_ex(x)
        logits = ins_model.classifier(features)
        logits_np = logits.cpu().numpy()

        # 2. Use Isolation Forest to detect anomalies
        predictions = iso_forest_model.predict(logits_np)  # 1 = normal, -1 = anomaly
        attention_weights = [0.001 if p == 1 else 0.999 for p in predictions]

        attention_tensor = torch.tensor(attention_weights).unsqueeze(0).to(bag_tensor.device)
        attention_tensor = attention_tensor / attention_tensor.sum()

        z = torch.mm(attention_tensor, features)
        class_prob = bag_classifier(z)
        return class_prob, None, z




# save pth
pth_path = f'./models/mnist_openmil.pth'

from torch.serialization import add_safe_globals
add_safe_globals([MIL])
mil_model = torch.load(pth_path, weights_only=False)

ins_model = insMLP(mil_model.feature_ex, mil_model.classifier, 3)
ins_model.eval()

import pandas as pd

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    '''Calculate MAVs'''

    mnist_train = bagDataset(
        train=True,
        epoch=0,
        num_bag=number_of_bags,
        bag_size=bag_size
    )

    mnist_calc = insDataset(
        train=True,
        epoch=100
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=10,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    calc_loader = torch.utils.data.DataLoader(
        mnist_calc,
        batch_size=10,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    # Step 2: Collect instance embeddings
    from sklearn.ensemble import IsolationForest

    def train_isolation_forest(features, labels):
        neg_logits = features[labels == 0]
        iso_forest = IsolationForest(contamination=0.1, random_state=0)
        iso_forest.fit(neg_logits)
        return iso_forest

    # Step 2: Collect instance embeddings
    features, labels = collect_instance_embeddings(ins_model, calc_loader)
    iso_forest_model = train_isolation_forest(features, labels)



    # Step 3: Fit OpenMax
    means, weibulls = fit_openmax(features, labels)
    bag_means, bag_weibulls = collect_bag_embeddings(mil_model, train_loader)
    


#This code is for step # 03


    '''Test Loader With Unknown Bags'''

    mnist_test = unk_bagDataset(
        train=False,
        epoch=0
    )

    test_loader = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=1,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

        
    correct_known = 0
    incorrect_known = 0
    correct_unknown = 0
    incorrect_unknown = 0

    y_true = []
    y_pred = []


    #TestLoader with unknown bags
    for bag_tensor, bag_label in test_loader:
        bag_tensor = bag_tensor.to(device)
        bag_label = bag_label.item()

        class_prob, unk_prob, z = predict_bag_with_modified_attention(
            bag_tensor,
            ins_model,
            iso_forest_model,
            mil_model.classifier
        )
        
        logits_np = class_prob.cpu().numpy().squeeze()
        bag_pred_label, bag_p_unknown = apply_openmax(logits_np, bag_means, bag_weibulls, threshold=2/3)



        true_bag_label = bag_label

        y_pred.append(bag_pred_label)
        y_true.append(true_bag_label)

        # Evaluation counters
        if true_bag_label == 3:
            if bag_pred_label == 3:
                correct_unknown += 1
            else:
                incorrect_unknown += 1
        else:
            if bag_pred_label == true_bag_label:
                correct_known += 1
            else:
                incorrect_known += 1

    total_known = correct_known + incorrect_known
    total_unknown = correct_unknown + incorrect_unknown
    total = total_known + total_unknown

    print(f"\nEvaluation Summary:")
    print(f"Known instance predictions: {total_known}")
    print(f" - Correct:   {correct_known}")
    print(f" - Incorrect: {incorrect_known}")

    print(f"Unknown instance predictions: {total_unknown}")
    print(f" - Correct:   {correct_unknown}")
    print(f" - Incorrect: {incorrect_unknown}")

    print(f"Overall Accuracy: {(correct_known + correct_unknown) / total:.2%}")
    
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # Display it
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2", "Unknown"])
    disp.plot(cmap='Blues', xticks_rotation=45)

    plt.title("Bag-Level Confusion Matrix (OpenMax)")
    plt.tight_layout()
    plt.show()


 

    