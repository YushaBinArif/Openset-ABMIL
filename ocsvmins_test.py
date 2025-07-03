# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import insDataset

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class IDs
class_n = 0  # negative
class_p1 = 1
class_p2 = 2
class_u = 3

# Model definition
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

# Collect embeddings
def collect_instance_embeddings(model, data_loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for input_tensor, label_tensor in data_loader:
            images = input_tensor[0]
            lbls = label_tensor[0]
            _, instance_feats = model(images)
            features.append(instance_feats.cpu())
            labels.append(lbls.cpu().view(-1))  # Ensures 1D even if scalar

    all_features = torch.cat(features, dim=0).numpy()
    all_labels = torch.cat(labels, dim=0).numpy()
    return all_features, all_labels
def collect_instance_embeddings(model, data_loader):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for input_tensor, label_tensor in data_loader:
            images = input_tensor[0]            # Shape: (num_instances, 784)
            instance_labels = label_tensor[0]   # Shape: (num_instances,)

            _, instance_feats = model(images)

            features.append(instance_feats.cpu())                 # (num_instances, D)
            labels.append(instance_labels.cpu().view(-1))         # (num_instances,)

    all_features = torch.cat(features, dim=0).numpy()  # shape: (total_instances, D)
    all_labels = torch.cat(labels, dim=0).numpy()      # shape: (total_instances,)

    return all_features, all_labels

# Load trained MIL model
pth_path = './models/mnist_openmil.pth'
from torch.serialization import add_safe_globals
add_safe_globals([MIL])
mil_model = torch.load(pth_path, weights_only=False)
ins_model = insMLP(mil_model.feature_ex, mil_model.classifier, 3)
ins_model.eval()

if __name__ == '__main__':
    # Training set for feature extraction
    train_set = insDataset(train=True, epoch=100)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)

    # Extract instance embeddings
    features, labels = collect_instance_embeddings(ins_model, train_loader)

    # Filter negative class only
    negative_features = features[labels == class_n]

    # Scale features and train OC-SVM
    scaler = StandardScaler()
    negative_features_scaled = scaler.fit_transform(negative_features)
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    ocsvm.fit(negative_features_scaled)

    # Inference
    test_set = insDataset(train=False, epoch=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    y_true = []
    y_pred = []
    correct_known = 0
    incorrect_known = 0
    correct_unknown = 0
    incorrect_unknown = 0

    for input_tensor, label_tensor in test_loader:
        image = input_tensor[0]
        true_instance_labels = label_tensor[0]
        _, feats = ins_model(image)
        feats_np = feats.detach().numpy()

        for i in range(feats_np.shape[0]):
            feat = feats_np[i].reshape(1, -1)
            feat_scaled = scaler.transform(feat)
            prediction = ocsvm.predict(feat_scaled)
            pred_label = 0 if prediction[0] == 1 else 3

            true_label = true_instance_labels[i].item()
            y_true.append(0 if true_label == 0 else 3)
            y_pred.append(pred_label)

            if true_label == 0:
                if pred_label == 0:
                    correct_known += 1
                else:
                    incorrect_known += 1
            else:
                if pred_label == 3:
                    correct_unknown += 1
                else:
                    incorrect_unknown += 1

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

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative (0)", "Positive/Unknown (3)"])
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Instance-Level Confusion Matrix (One-Class SVM)")
    plt.tight_layout()
    plt.show()