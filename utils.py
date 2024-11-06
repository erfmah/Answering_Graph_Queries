import sys
import os
import torch
import random
import math
import csv
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, recall_score, \
    precision_score, precision_recall_curve
from sklearn.metrics import auc as PRAUC
from numpy import argmax
import copy
import scipy.sparse as sp
import numpy as np
from scipy import sparse
import dgl
from scipy.stats import multivariate_normal
import torch.nn.functional as F
from sklearn.metrics import f1_score





def get_metrics(target_edges, org_adj, reconstructed_adj):
    reconstructed_adj =  sparse.csr_matrix(torch.sigmoid(reconstructed_adj).detach().numpy())
    org_adj = sparse.csr_matrix(org_adj)
    prediction = []
    true_label = []
    counter = 0
    for edge in target_edges:
        prediction.append(reconstructed_adj[edge[0], edge[1]])
        prediction.append(reconstructed_adj[edge[1], edge[0]])
        true_label.append(org_adj[edge[0], edge[1]])
        true_label.append(org_adj[edge[1], edge[0]])

    pred = np.array(prediction)
    
    
    precision, recall, thresholds = precision_recall_curve(true_label, pred)
    filter = recall >= 0.8  # or any other recall level you deem necessary
    best_threshold = thresholds[np.argmax(precision[filter])] if any(filter) else 0.5
    Threshold = best_threshold
    pr_auc = PRAUC(recall, precision)

    # fscore = (2 * precision * recall) / (precision + recall)
    # ix = argmax(fscore)
    # Threshold = thresholds[ix]
    # Threshold = 0.5
    # thresholds = np.append(thresholds, 1)
    # acc = [accuracy_score(true_label, prediction >= t) for t in thresholds]
    
    pred[pred > Threshold] = 1.0
    pred[pred < Threshold] = 0.0
    pred = pred.astype(int)


    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)

    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:] # dividing by 5 to get top 20%
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
    
    
    return auc, acc, ap, precision, recall, HR, np.max(thresholds)



def roc_auc_single(prediction, true_label):
    pred = np.array(prediction)
    pred[pred > .5] = 1
    pred[pred < .5] = 0
    pred = pred.astype(int)
    # pred = prob_to_one_hot(pred)

    precision = precision_score(y_pred=pred, y_true=true_label)
    recall = recall_score(y_pred=pred, y_true=true_label)
    auc = roc_auc_score(y_score=prediction, y_true=true_label)
    acc = accuracy_score(y_pred=pred, y_true=true_label, normalize=True)
    ap = average_precision_score(y_score=prediction, y_true=true_label)
    hr_ind = np.argpartition(np.array(prediction), -1*len(pred)//5)[-1*len(pred)//5:] # dividing by 5 to get top 20%
    HR = precision_score(y_pred=np.array(pred)[hr_ind], y_true=np.array(true_label)[hr_ind])
    pred = np.array(prediction)
    
    return auc, acc, ap, precision, recall, HR

def roc_auc_estimator_labels(re_labels, labels, org_labels):
    prediction = []
    true_label = []

    for i in range(len(labels)):
        prediction.append(re_labels[i].detach().numpy())
        true_label.append(labels[i].detach().numpy())
    prediction = np.array(prediction)
    true_label = np.array(true_label)
    num_classes = true_label.shape[1]  # Number of classes
    # pred = prediction
    # pred =
    # pred[pred > .5] = 1.0
    # pred[pred < .5] = 0.0
    # pred = pred.astype(int)
    pred = prob_to_one_hot(prediction)

    precision = precision_score(y_pred=pred, y_true=true_label, average="weighted")
    recall = recall_score(y_pred=pred, y_true=true_label, average="weighted")

    roc_auc_scores = []
    seen_classes = 0

    for i in range(num_classes):
        # Calculate ROC-AUC for each class
        y_true = torch.from_numpy(true_label[:, i])
        y_pred = torch.from_numpy(prediction[:, i])
        y_true = torch.cat([y_true, torch.tensor([0])])
        y_pred = torch.cat([y_pred, torch.tensor([0])])
        if len(y_true.nonzero()) > 0:
            seen_classes += 1
            roc_auc = roc_auc_score(y_true, y_pred)
            roc_auc_scores.append(roc_auc)

    average_roc_auc = sum(roc_auc_scores) / seen_classes


    acc = accuracy_score(y_pred=pred, y_true=true_label)
    ap = average_precision_score(y_score=prediction, y_true=true_label)

    f1_score_macro = f1_score(true_label, pred, average ="macro")
    return average_roc_auc, acc, ap, precision, recall, f1_score_macro

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = 1
    return ret



def run_network(feats, adj, labels, model, targets, sampling_method, is_prior):
    adj = sparse.csr_matrix(adj)
    graph_dgl = dgl.from_scipy(adj)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops  
    std_z, m_z, z, re_adj, reconstructed_feat, reconstructed_labels = model(graph_dgl, feats, labels, targets,                                                             sampling_method, is_prior, train=False)
    return std_z, m_z, z, re_adj, reconstructed_feat, reconstructed_labels


def get_pdf(mean_p, std_p, mean_q, std_q, z, targets):

    pdf_all_z_p = 0
    pdf_all_z_q = 0
    for i in targets:
        # TORCH
        cov_p = np.diag(std_p.detach().numpy()[i] ** 2)
        dist_p = torch.distributions.multivariate_normal.MultivariateNormal(mean_p[i], torch.tensor(cov_p))
        pdf_all_z_p += dist_p.log_prob(z[i]).detach().numpy()

        cov_q = np.diag(std_q.detach().numpy()[i] ** 2)
        dist_q = torch.distributions.multivariate_normal.MultivariateNormal(mean_q[i], torch.tensor(cov_q))
        pdf_all_z_q += dist_q.log_prob(z[i]).detach().numpy()
    return pdf_all_z_p, pdf_all_z_q

def weight_labels(labels):
    n_samples = labels.shape[0]
    labels_ind = torch.argmax(torch.from_numpy(labels), dim=1)
    class_counts = torch.bincount(labels_ind)
    class_weights = []
    num_classes = labels.shape[1]
    for i in range(0,num_classes):
        class_weights.append(n_samples/(class_counts[i]*num_classes))
    return torch.tensor(class_weights)
    # labels = torch.argmax(torch.from_numpy(labels), dim=1)
    # # labels = torch.from_numpy(labels)
    # class_counts = torch.bincount(labels)
    #
    # # Calculate the total number of samples
    # total_samples = len(labels)
    #
    # # Calculate class frequencies (class_counts / total_samples)
    # class_frequencies = class_counts.float() / total_samples
    #
    # # Calculate inverse class frequencies to use as class weights
    # class_weights = 1.0 / class_frequencies
    # class_weights /= class_weights.sum()


def weight_edges(labels):
    # labels = torch.from_numpy(labels)
    n_samples = labels.shape[0]*labels.shape[1]
    # labels_ind = torch.argmax(torch.from_numpy(labels), dim=1)
    class_counts = torch.tensor([(labels.shape[0] ** 2 - torch.sum(labels)),torch.sum(labels) ])
    class_weights = []
    num_classes = 2
    for i in range(0,num_classes):
        class_weights.append(n_samples/(class_counts[i]*num_classes))
    return torch.tensor(class_weights)

def test(test_edges, org_adj, run_network, features, labels, inductive_model, targets, sampling_method):
    adj_list_copy = copy.deepcopy(org_adj)
    for i, j in test_edges:
        adj_list_copy[i][j] = 0

    std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features,
                                                                                                adj_list_copy,
                                                                                                labels,
                                                                                                inductive_model,
                                                                                                targets,
                                                                                                sampling_method,
                                                                                                is_prior=True)
    re_adj_prior_sig = torch.sigmoid(re_adj_prior)
    re_label_prior_sig = torch.sigmoid(re_prior_labels)
    pred_single_link = []
    true_single_link = []
    pred_single_label = []
    true_single_label = []
    for i,j in test_edges:
        pred_single_link.append(re_adj_prior_sig[i][j].detach().numpy())
        true_single_link.append(org_adj[i][j])
        pred_single_label.append(re_label_prior_sig[i])
        true_single_label.append(labels[i])
    auc, val_acc, val_ap, precision, recall, HR = roc_auc_single(pred_single_link, true_single_link)
    auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = roc_auc_estimator_labels(pred_single_label, true_single_label,
                                                                                   labels)
    return auc, val_acc, val_ap, precision, recall, HR, auc_l, acc_l, ap_l, precision_l, recall_l, F1_score








