# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import csv
import os
import time
import sys
import random
import scipy.spatial.distance as spd
import libmr

# Import necessary libraries
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#import seaborn as sns

def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []

        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


#TODO: Formulate equation for prob_unknowns
def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)
    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    print(ranked_list)
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights


    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)

            #TODO: Formulate equation for openmax_prob, softmax_prob
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob

def openmaxatt(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = bag_size
    alpha = 10
    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    print(ranked_list)
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights


    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)

            #TODO: Formulate equation for openmax_prob, softmax_prob
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists_bfr(train_class_num,trainloader,device,model):
    scores = [[] for _ in range(train_class_num)]
    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in trainloader:
            #inputs, targets = inputs.to(device), targets.to(device)

            #Step: 01 Check if the targets[0] is equals to class_n
            #Step: 02 If targets[0] is equals to class_n, loop 10 times to pass input_tensor[0][i] to model()
            if targets[0] in class_n:
              for i in range(10):
                outputs, attention = model(inputs[0][i].unsqueeze(0).unsqueeze(0))
                for score, t in zip(outputs, targets):
                    if torch.argmax(score) == t:
                        scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
            else:
                outputs, attention = model(inputs[0].unsqueeze(dim=0))
                for score, t in zip(outputs, targets):
                    if torch.argmax(score) == t:
                        scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists



def compute_train_score_and_mavs_and_dists(train_class_num,trainloader,device,model):
    scores = [[] for _ in range(train_class_num)]
    model.eval()
    with torch.no_grad():
        for inputs, targets, _ in trainloader:
            #inputs, targets = inputs.to(device), targets.to(device)

            # this must cause error for cifar
            outputs, attention = model(inputs[0]) # 1 x 3

            for score, t in zip(outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                # only considering the correctly classified example during the training time
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    print(scores.shape)
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    print(mavs.shape)
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return scores, mavs, dists

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, \
    classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


class Evaluation(object):
    """Evaluation class based on python list"""
    def __init__(self, predict, label,prediction_scores = None):
        self.predict = predict
        self.label = label
        self.prediction_scores = prediction_scores

        self.accuracy = self._accuracy()

        # filename = "test-accuracy.txt"
        # print('Accuracy:', f"%.3f" % self.accuracy)
        # append_result_to_file(filename, 'Accuracy:' + f"%.3f" % self.accuracy )

        # self.f1_measure = self._f1_measure()
        # prqint('F1-measure:', f'{self.f1_measure:.3f}')
        # self.f1_macro = self._f1_macro()
        # print('F1-macro:', f'{self.f1_macro:.3f}')
        # self.f1_macro_weighted = self._f1_macro_weighted()
        # print('F1-macro (weighted):', f'{self.f1_macro_weighted:.3f}')
        # self.precision, self.recall = self._precision_recall(average='micro')
        # print('precision:', f'{self.precision:.3f}')
        # print('precision (macro):', f'{self.precision_macro:.3f}')
        # print('precision (weighted):', f'{self.precision_weighted:.3f}')
        # self.precision_macro, self.recall_macro = self._precision_recall(average='macro')
        # print('recall:', f'{self.recall:.3f}')
        # print('recall (macro):', f'{self.recall_macro:.3f}')
        # self.precision_weighted, self.recall_weighted = self._precision_recall(average='weighted')
        # print('recall (weighted):', f'{self.recall_weighted:.3f}')
        # self.confusion_matrix = self._confusion_matrix()
        if self.prediction_scores is not None:
            self.area_under_roc = self._area_under_roc(prediction_scores)
            # print('Area under ROC curve (with 100% confidence in prediction):', f'{self.area_under_roc():.3f}')
            #print('Area under ROC curve (with 100% confidence in prediction):', f'{self.area_under_roc:.3f}')


    def _accuracy(self) -> float:
        """
        Returns the accuracy score of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        correct = (np.array(self.predict) == np.array(self.label)).sum()
        return float(correct)/float(len(self.predict))

    def _f1_measure(self) -> float:
        """
        Returns the F1-measure with a micro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='micro')

    def _f1_macro(self) -> float:
        """
        Returns the F1-measure with a macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='macro')

    def _f1_macro_weighted(self) -> float:
        """
        Returns the F1-measure with a weighted macro average of the labels and predictions.
        :return: float
        """
        assert len(self.predict) == len(self.label)
        return f1_score(self.label, self.predict, average='weighted')

    def _precision_recall(self, average) -> (float, float):
        """
        Returns the precision and recall scores for the label and predictions. Observes the average type.

        :param average: string, [None (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’].
            For explanations of each type of average see the documentation for
            `sklearn.metrics.precision_recall_fscore_support`
        :return: float, float: representing the precision and recall scores respectively
        """
        assert len(self.predict) == len(self.label)
        precision, recall, _, _ = precision_recall_fscore_support(self.label, self.predict, average=average)
        return precision, recall

    def _area_under_roc(self, prediction_scores: np.array = None, multi_class='ovo') -> float:
        """
        Area Under Receiver Operating Characteristic Curve

        :param prediction_scores: array-like of shape (n_samples, n_classes). The multi-class ROC curve requires
            prediction scores for each class. If not specified, will generate its own prediction scores that assume
            100% confidence in selected prediction.
        :param multi_class: {'ovo', 'ovr'}, default='ovo'
            'ovo' computes the average AUC of all possible pairwise combinations of classes.
            'ovr' Computes the AUC of each class against the rest.
        :return: float representing the area under the ROC curve
        """
        label, predict = self.label, self.predict
        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        one_hot_encoder.fit(np.array(label).reshape(-1, 1))
        true_scores = one_hot_encoder.transform(np.array(label).reshape(-1, 1))
        if prediction_scores is None:
            prediction_scores = one_hot_encoder.transform(np.array(predict).reshape(-1, 1))
        # assert prediction_scores.shape == true_scores.shape
        return roc_auc_score(true_scores, prediction_scores, multi_class=multi_class)

    def _confusion_matrix(self, normalize=None) -> np.array:
        """
        Returns the confusion matrix corresponding to the labels and predictions.

        :param normalize: {‘true’, ‘pred’, ‘all’}, default=None.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        :return:
        """
        assert len(self.predict) == len(self.label)
        return confusion_matrix(self.label, self.predict, normalize=normalize)

    def plot_confusion_matrix(self, labels: [str] = None, normalize=None, ax=None, savepath=None) -> None:
        """

        :param labels: [str]: label names
        :param normalize: {‘true’, ‘pred’, ‘all’}, default=None.
            Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
        :param ax: matplotlib.pyplot axes to draw the confusion matrix on. Will generate new figure/axes if None.
        :return:
        """
        conf_matrix = self._confusion_matrix(normalize)  # Evaluate the confusion matrix
        display = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)  # Generate the confusion matrix display

        # Formatting for the plot
        if labels:
            xticks_rotation = 'vertical'
        else:
            xticks_rotation = 'horizontal'

        display.plot(include_values=True, cmap=plt.cm.get_cmap('Blues'), xticks_rotation=xticks_rotation, ax=ax)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, bbox_inches='tight', dpi=200)
        plt.close()
