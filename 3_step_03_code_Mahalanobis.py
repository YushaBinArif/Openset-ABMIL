# Updated Mahalanobis-based attention integration for OpenMIL

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt

from dataset import bagDataset, insDataset
from dataset_unk_bags import bagDataset as unk_bagDataset

# ---------------------- Model Definitions ----------------------

class MIL(nn.Module):
    def __init__(self, n_class):
        super(MIL, self).__init__()
        self.n_class = n_class
        self.feature_ex = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 16)
        )
        self.attention = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        self.classifier = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, n_class))

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
        self.feature_ex = feature_ex
        self.classifier = classifier
        self.n_class = n_class

    def forward(self, x):
        x = x.squeeze(0)
        h = self.feature_ex(x)
        class_prob = self.classifier(h).reshape(-1, self.n_class)
        return class_prob, h

# ---------------------- Mahalanobis Scorer ----------------------

class MahalanobisAnomalyScorer:
    def __init__(self):
        self.mean = None
        self.cov_inv = None

    def fit(self, features, labels):
        # Use only class 0 (negative class) for anomaly fitting
        features_class_0 = features[labels == 0]
        self.mean = np.mean(features_class_0, axis=0)
        self.cov_inv = np.linalg.inv(np.cov(features_class_0.T))

    def score(self, feature):
        diff = feature - self.mean
        return np.sqrt(np.dot(np.dot(diff, self.cov_inv), diff.T))

# ---------------------- Embedding Collection ----------------------

def collect_instance_embeddings(model, data_loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, true_labels in data_loader:
            x = images.to(device).squeeze(0)
            feats = model.feature_ex(x)
            logits = model.classifier(feats)
            preds = torch.argmax(logits, dim=1)
            correct = preds == true_labels.to(device)
            features.append(feats[correct].cpu())
            labels.append(true_labels[correct].cpu())
    return torch.cat(features).numpy(), torch.cat(labels).numpy()

# ---------------------- Prediction Function ----------------------

def predict_bag_with_mahalanobis_attention(bag_tensor, ins_model, scorer, scaler, bag_classifier):
    with torch.no_grad():
        x = bag_tensor.squeeze(0).to(device)
        features = ins_model.feature_ex(x)
        features_np = features.cpu().numpy()
        features_scaled = scaler.transform(features_np)

        distances = np.array([scorer.score(f) for f in features_scaled])
        max_d = distances.max() + 1e-8
        attention_weights = 1.0 - (distances / max_d)

        attention_tensor = torch.tensor(attention_weights).unsqueeze(0).to(device)
        attention_tensor = attention_tensor / attention_tensor.sum()
        z = torch.mm(attention_tensor.float(), features.float())
        class_prob = bag_classifier(z)
        return class_prob, attention_tensor, z

# ---------------------- Main ----------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model_path = './models/mnist_openmil.pth'
mil_model = torch.load(model_path, weights_only=False)
ins_model = insMLP(mil_model.feature_ex, mil_model.classifier, 3).eval()

# Load datasets
calc_loader = DataLoader(insDataset(train=True, epoch=100), batch_size=10)
train_loader = DataLoader(bagDataset(train=True, epoch=0, num_bag=1000, bag_size=10), batch_size=10)
test_loader = DataLoader(unk_bagDataset(train=False, epoch=0), batch_size=1)

# Step 1: Instance feature extraction
features, labels = collect_instance_embeddings(ins_model, calc_loader)
scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

# Step 2: Fit Mahalanobis scorer
mahalanobis_scorer = MahalanobisAnomalyScorer()
mahalanobis_scorer.fit(features_scaled, labels)

# Step 3: Inference
correct_known, incorrect_known = 0, 0
correct_unknown, incorrect_unknown = 0, 0
y_true, y_pred = [], []

for bag_tensor, bag_label in test_loader:
    bag_tensor = bag_tensor.to(device)
    bag_label = bag_label.item()

    class_prob, _, _ = predict_bag_with_mahalanobis_attention(
        bag_tensor, ins_model, mahalanobis_scorer, scaler, mil_model.classifier
    )

    logits_np = class_prob.cpu().numpy().squeeze()
    bag_pred_label = np.argmax(logits_np)
    if bag_pred_label == 3 or logits_np.max() < 0.5:
        bag_pred_label = 3  # threshold-based rejection

    y_pred.append(bag_pred_label)
    y_true.append(bag_label)

    if bag_label == 3:
        if bag_pred_label == 3: correct_unknown += 1
        else: incorrect_unknown += 1
    else:
        if bag_pred_label == bag_label: correct_known += 1
        else: incorrect_known += 1

# Summary
print("\nEvaluation Summary:")
print(f"Known predictions: {correct_known}/{correct_known + incorrect_known}")
print(f"Unknown predictions: {correct_unknown}/{correct_unknown + incorrect_unknown}")
acc = (correct_known + correct_unknown) / (correct_known + correct_unknown + incorrect_known + incorrect_unknown)
print(f"Overall Accuracy: {acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg(0)", "Pos(1)", "Pos(2)", "Unknown"]).plot(cmap='Blues')
plt.title("Mahalanobis + OpenMax Confusion Matrix")
plt.tight_layout()
plt.show()