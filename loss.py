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

from utils import *

# objective Function
def  optimizer_VAE (lambda_1,lambda_2, lambda_3, true_labels, reconstructed_labels, loss_type, pred, reconstructed_feat, labels, x, norm_feat, pos_weight_feat,  std_z, mean_z, num_nodes, pos_weight, norm, indexes, trainID):
    val_poterior_cost = 0
    w_l = weight_labels(true_labels)
    # pos_weight = weight_edges(labels)
    # posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)
    posterior_cost_edges = \
        (norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight, reduction = 'none')).mean()

    posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat, x, pos_weight=pos_weight_feat)

    posterior_cost_classes = F.cross_entropy(reconstructed_labels, (torch.tensor(true_labels).to(torch.float64)))
    z_kl = (-0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * torch.log(std_z) - mean_z.pow(2) - (std_z).pow(2), dim=1))

    acc = (torch.sigmoid(pred).round() == labels).sum() / float(pred.shape[0] * pred.shape[1])
    adj_shape = labels.shape[0]*labels.shape[1]
    features_shape = x.shape[0]*x.shape[1]
    labels_shape = reconstructed_labels.shape[0]*reconstructed_labels.shape[1]

    if loss_type == "0":
        posterior_cost = posterior_cost_classes
    elif loss_type == "1":
        posterior_cost = lambda_1 * posterior_cost_edges + lambda_2 * posterior_cost_features + lambda_3 * posterior_cost_classes
    elif loss_type == "2":
        posterior_cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes
    elif loss_type == "3":
        posterior_cost = posterior_cost_edges
    elif loss_type == "4":
        posterior_cost = posterior_cost_edges+posterior_cost_classes
    return z_kl, posterior_cost,posterior_cost_edges ,posterior_cost_features , posterior_cost_classes, acc, val_poterior_cost, posterior_cost_edges, posterior_cost_features

