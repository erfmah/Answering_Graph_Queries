import sys
import os
import torch
import random
import math
import csv
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score, recall_score, \
    precision_score, precision_recall_curve
from numpy import argmax
import copy
import scipy.sparse as sp
import numpy as np
from scipy import sparse
import dgl
from scipy.stats import multivariate_normal
import torch.nn.functional as F
from sklearn.metrics import f1_score



# objective Function
def  optimizer_VAE (lambda_1,lambda_2, lambda_3, true_labels, reconstructed_labels, loss_type, pred, reconstructed_feat, labels, x, norm_feat, pos_weight_feat,  std_z, mean_z, num_nodes, pos_weight, norm):
    # val_poterior_cost = 0
    # w_l = weight_labels(true_labels)
    # posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)
    # posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat, x, pos_weight=pos_weight_feat)
    # posterior_cost_classes = F.cross_entropy(reconstructed_labels, (torch.tensor(true_labels).to(torch.float64)), weight=w_l)
    # z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))
    #
    # shape_adj = pred.shape[0]*pred.shape[1]
    # shape_feat = x.shape[0]*x.shape[1]
    # shape_labels = true_labels.shape[0]
    #
    # total = shape_adj+shape_feat+shape_labels
    #
    # acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    # ones_x = x.nonzero().shape[0]
    # ones_adj = labels.nonzero().shape[0]
    # if loss_type == "0":
    #     posterior_cost = posterior_cost_edges
    # elif loss_type == "1":
    #     posterior_cost = lambda_1 * posterior_cost_edges + (1-lambda_1) * posterior_cost_features
    # elif loss_type == "2":
    #     posterior_cost = lambda_1 * (1 / ones_adj) * posterior_cost_edges + (1-lambda_1) * (1 / ones_x) * posterior_cost_features
    # elif loss_type == "3":
    #     posterior_cost = lambda_1 * (ones_x / ones_adj) * posterior_cost_edges + (1-lambda_1) * (ones_adj / ones_x) * posterior_cost_features
    # elif loss_type == "4":
    #     posterior_cost = lambda_1 * (1 / (labels.shape[0]*labels.shape[0])) * posterior_cost_edges + (1-lambda_1) * (1 / (x.shape[0]*x.shape[1])) * posterior_cost_features
    # elif loss_type == "5":
    #     # posterior_cost = lambda_1 *(1/shape_adj)* posterior_cost_edges + lambda_2 *(1/shape_labels)*posterior_cost_classes + lambda_3 *(1/shape_feat)*posterior_cost_features
    #     # posterior_cost = (shape_feat/shape_adj)* lambda_1 * posterior_cost_edges + lambda_2 * posterior_cost_classes + (shape_adj/shape_feat)*lambda_3  * posterior_cost_features
    #     # posterior_cost = (shape_feat / shape_adj)  * posterior_cost_edges +   posterior_cost_classes + (shape_adj / shape_feat)  * posterior_cost_features
    #     # posterior_cost = posterior_cost_edges + posterior_cost_classes + posterior_cost_features
    #     posterior_cost = posterior_cost_edges + posterior_cost_features
    #
    # elif loss_type == "6":
    #     posterior_cost = lambda_1 * (ones_adj / ones_x) * posterior_cost_edges + (1-lambda_1) * (ones_x / ones_adj) * posterior_cost_features
    # elif loss_type == "7":
    #     posterior_cost = lambda_1 * ((x.shape[0] * x.shape[1]) / (labels.shape[0] * labels.shape[0])) * posterior_cost_edges +  (1-lambda_1) * (
    #             (labels.shape[0] * labels.shape[1]) / (x.shape[0] * x.shape[1])) * posterior_cost_features
    #     # posterior_cost = ((x.shape[0] * x.shape[1]) / (labels.shape[0] * labels.shape[1])) * posterior_cost_edges + (
    #     #                          (labels.shape[0] * labels.shape[1]) / (x.shape[0] * x.shape[1])) * posterior_cost_features
    # elif loss_type == "8":
    #     posterior_cost = lambda_1 *((x.shape[0] * x.shape[1]) / (labels.shape[0] * labels.shape[1])) * posterior_cost_edges + (1-lambda_1) *(
    #             (labels.shape[0] * labels.shape[1]) / (x.shape[0] * x.shape[1])) * posterior_cost_features
    # elif loss_type == "9":
    #     posterior_cost = posterior_cost_edges+posterior_cost_features
    #
    # else:
    #     posterior_cost = lambda_1 * ((labels.shape[0] * labels.shape[1]) / (x.shape[0] * x.shape[1])) * posterior_cost_edges + (1-lambda_1) * (
    #             (x.shape[0] * x.shape[1]) / (labels.shape[0] * labels.shape[1])) * posterior_cost_features

    val_poterior_cost = 0
    w_l = weight_labels(true_labels)
    # pos_weight = weight_edges(labels)
    posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)
    posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat, x, pos_weight=pos_weight_feat)
    # MSE_loss = torch.nn.MSELoss()
    # posterior_cost_features = norm_feat * MSE_loss(reconstructed_feat, x)
    posterior_cost_classes = F.cross_entropy(reconstructed_labels, (torch.tensor(true_labels).to(torch.float64)), weight=w_l)

    z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))

    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    adj_shape = labels.shape[0]*labels.shape[1]
    features_shape = x.shape[0]*x.shape[1]
    labels_shape = reconstructed_labels.shape[0]*reconstructed_labels.shape[1]

    if loss_type == "0":
        posterior_cost = posterior_cost_classes
    elif loss_type == "1":
        posterior_cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes
    elif loss_type == "2":
        posterior_cost = (adj_shape/features_shape) * posterior_cost_edges + (features_shape/adj_shape) * posterior_cost_features
    elif loss_type == "3":
        posterior_cost = posterior_cost_edges + posterior_cost_classes
    return z_kl, posterior_cost,posterior_cost_edges ,posterior_cost_features , posterior_cost_classes, acc, val_poterior_cost, posterior_cost_edges, posterior_cost_features



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
    fscore = (2 * precision * recall) / (precision + recall)
    ix = argmax(fscore)
    Threshold = thresholds[ix]
    Threshold = 0.5
    
    
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
    
    
    return auc, acc, ap, precision, recall, HR



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



    for i in range(num_classes):
        # Calculate ROC-AUC for each class
        y_true = torch.from_numpy(true_label[:, i])
        y_pred = torch.from_numpy(prediction[:, i])
        y_true = torch.cat([y_true, torch.tensor([0])])
        y_pred = torch.cat([y_pred, torch.tensor([0])])
        if len(y_true.nonzero())>0:
            roc_auc = roc_auc_score(y_true, y_pred)
            roc_auc_scores.append(roc_auc)

    average_roc_auc = sum(roc_auc_scores) / num_classes


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
    n_samples = labels.shape[0]
    labels_ind = torch.argmax(torch.from_numpy(labels), dim=1)
    class_counts = torch.bincount(labels_ind)
    class_weights = []
    num_classes = labels.shape[1]
    for i in range(0,num_classes):
        class_weights.append(n_samples/(class_counts[i]*num_classes))
    return torch.tensor(class_weights)

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








