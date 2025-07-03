# Full script with OpenMax replaced by OC-SVM for bag classification

import os
import sys
import time
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import bagDataset, insDataset
from dataset_unk_bags import bagDataset as unk_bagDataset

# Set seed and device
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_class_num = 3
class_n, class_p1, class_p2, class_u = 0, 1, 2, 3
number_of_bags = 1000
bag_size = 10

# Define MIL and instance models
class MIL(nn.Module):
    def __init__(self, n_class):
        super(MIL, self).__init__()
        self.n_class = n_class
        self.feature_ex = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 16)
        )
        self.attention = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, n_class)
        )

    def forward(self, input):
        x = input.squeeze(0)
        h = self.feature_ex(x)
        a = self.attention(h)
        a_t = torch.transpose(a, 1, 0)
        a_n = F.softmax(a_t, dim=1)
        z = torch.mm(a_n, h)
        class_prob = self.classifier(z).reshape(1, self.n_class)
        return class_prob, a_n

class insMLP(nn.Module):
    def __init__(self, feature_ex, classifier, n_class):
        super(insMLP, self).__init__()
        self.n_class = n_class
        self.feature_ex = feature_ex
        self.classifier = classifier

    def forward(self, input):
        x = input.squeeze(0)
        h = self.feature_ex(x)
        class_prob = self.classifier(h).reshape(-1, self.n_class)
        return class_prob, h

# Data collection functions
def collect_instance_embeddings(model, data_loader):
    model.eval()
    features, preds = [], []
    with torch.no_grad():
        for images, true_labels in data_loader:
            images = images.to(device)
            true_labels = true_labels.to(device)
            instance_logits, _ = model(images)
            predicted_labels = torch.argmax(instance_logits, dim=1)
            correct_mask = (predicted_labels == true_labels)
            correct_feats = instance_logits[correct_mask]
            correct_labels = true_labels[correct_mask]
            features.append(correct_feats.cpu())
            preds.append(correct_labels.cpu())
    return torch.cat(features).numpy(), torch.cat(preds).numpy()

def collect_bag_logits(model, loader):
    model.eval()
    class_feats = defaultdict(list)
    with torch.no_grad():
        for bags, labels in loader:
            for i in range(bags.size(0)):
                x = bags[i].unsqueeze(0)
                y = labels[i].item()
                logits, _ = model(x)
                pred = torch.argmax(logits, dim=1).item()
                logit = logits.squeeze().cpu().numpy()
                if y == class_n:
                    class_feats[class_n].append(logit)
                elif y in [class_p1, class_p2]:
                    if pred == y:
                        class_feats[y].append(logit)
                    elif pred == class_n:
                        class_feats[class_n].append(logit)
    return class_feats

# Load models
pth_path = './models/mnist_openmil.pth'
mil_model = torch.load(pth_path, weights_only=False)
ins_model = insMLP(mil_model.feature_ex, mil_model.classifier, 3)
ins_model.eval()

# Load datasets
mnist_train = bagDataset(train=True, epoch=0, num_bag=number_of_bags, bag_size=bag_size)
mnist_calc = insDataset(train=True, epoch=100)
train_loader = DataLoader(mnist_train, batch_size=10, shuffle=True)
calc_loader = DataLoader(mnist_calc, batch_size=10, shuffle=True)

features, labels = collect_instance_embeddings(ins_model, calc_loader)
class_feats = collect_bag_logits(mil_model, train_loader)

# Train One-Class SVMs
ocsvm_models = {}
for cls in class_feats:
    feats = np.stack(class_feats[cls])
    ocsvm = OneClassSVM(gamma='auto', nu=0.1)
    ocsvm.fit(feats)
    ocsvm_models[cls] = ocsvm

# Test loop
mnist_test = unk_bagDataset(train=False, epoch=0)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

y_true, y_pred = [], []
correct_known = incorrect_known = correct_unknown = incorrect_unknown = 0

def predict_with_ocsvm(z, models):
    scores = {cls: model.decision_function(z.detach().cpu().numpy())[0] for cls, model in models.items()}
    best_cls = max(scores, key=scores.get)
    return 3 if scores[best_cls] < 0 else best_cls

for bag_tensor, bag_label in test_loader:
    bag_tensor = bag_tensor.to(device)
    bag_label = bag_label.item()
    z,_ = mil_model(bag_tensor)
    bag_pred_label = predict_with_ocsvm(z, ocsvm_models)

    y_true.append(bag_label)
    y_pred.append(bag_pred_label)

    if bag_label == 3:
        if bag_pred_label == 3: correct_unknown += 1
        else: incorrect_unknown += 1
    else:
        if bag_pred_label == bag_label: correct_known += 1
        else: incorrect_known += 1

# Results
print("\nEvaluation Summary:")
print(f"Known instance predictions: {correct_known + incorrect_known}")
print(f" - Correct:   {correct_known}")
print(f" - Incorrect: {incorrect_known}")
print(f"Unknown instance predictions: {correct_unknown + incorrect_unknown}")
print(f" - Correct:   {correct_unknown}")
print(f" - Incorrect: {incorrect_unknown}")
print(f"Overall Accuracy: {(correct_known + correct_unknown) / (correct_known + incorrect_known + correct_unknown + incorrect_unknown):.2%}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1", "Class 2", "Unknown"])
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Bag-Level Confusion Matrix (OC-SVM)")
plt.tight_layout()
plt.show()
