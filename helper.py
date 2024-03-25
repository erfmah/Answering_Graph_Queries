#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:13:06 2021

@author: pnaddaf
"""

import sys
import os
import argparse

import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import random
from scipy import sparse
from dgl.nn.pytorch import GraphConv as GraphConv
from dataCenter import *
from utils import *
from models import *
import timeit
from torch.nn.functional import normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

def train_model(dataCenter, features, args, device):
    decoder = args.decoder_type
    encoder = args.encoder_type
    num_of_relations = args.num_of_relations  # diffrent type of relation
    num_of_comunities = args.num_of_comunities  # number of comunities
    batch_norm = args.batch_norm
    DropOut_rate = args.DropOut_rate
    encoder_layers = [int(x) for x in args.encoder_layers.split()]
    epoch_number = args.epoch_number
    subgraph_size = args.num_node
    lr = args.lr
    is_prior = args.is_prior
    targets = args.targets
    sampling_method = args.sampling_method
    ds = args.dataSet
    loss_type = args.loss_type


    original_adj_full= torch.FloatTensor(getattr(dataCenter, ds+'_adj_lists')).to(device)
    node_label_full= torch.FloatTensor(getattr(dataCenter, ds+'_labels')).to(device)


    # shuffling the data, and selecting a subset of it
    if subgraph_size == -1:
        subgraph_size = original_adj_full.shape[0]
    elemnt = min(original_adj_full.shape[0], subgraph_size)
    indexes = list(range(original_adj_full.shape[0]))
    np.random.shuffle(indexes)
    indexes = indexes[:elemnt]
    original_adj = original_adj_full[indexes, :]
    original_adj = original_adj[:, indexes]

    node_label = [np.array(node_label_full[i], dtype=np.float16) for i in indexes]
    features = features[indexes]
    number_of_classes = len(node_label_full[0])

    # Check for Encoder and redirect to appropriate function
    if encoder == "Multi_GCN":
        encoder_model = multi_layer_GCN(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)
        # encoder_model = multi_layer_GCN(in_feature=features.shape[1], latent_dim=num_of_comunities, layers=encoder_layers)

    elif encoder == "Multi_GAT":
        encoder_model = multi_layer_GAT(num_of_comunities , latent_dim=num_of_comunities, layers=encoder_layers)


    elif encoder == "Multi_GIN":
        encoder_model = multi_layer_GIN(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)

    else:
        raise Exception("Sorry, this Encoder is not Impemented; check the input args")

    # Check for Decoder and redirect to appropriate function

    if decoder == "ML_SBM":
        decoder_model = MultiLatetnt_SBM_decoder(num_of_relations, num_of_comunities, num_of_comunities, batch_norm, DropOut_rate=0.3)

    else:
        raise Exception("Sorry, this Decoder is not Impemented; check the input args")

    feature_encoder_model = feature_encoder(features.view(-1, features.shape[1]), num_of_comunities)
    # feature_encoder_model = MulticlassClassifier(num_of_comunities, features.shape[1])
    feature_decoder = feature_decoder_nn(features.shape[1], num_of_comunities)
    class_decoder = MulticlassClassifier(number_of_classes, num_of_comunities)


    trainId = getattr(dataCenter, ds + '_train')
    testId = getattr(dataCenter, ds + '_test')
    validId = getattr(dataCenter, ds + '_val')

    adj_train =  original_adj.cpu().detach().numpy()[trainId, :][:, trainId]
    adj_val = original_adj.cpu().detach().numpy()[validId, :][:, validId]

    feat_np = features.cpu().data.numpy()
    feat_train = feat_np[trainId, :]
    feat_val = feat_np[validId, :]

    labels_np = np.array(node_label, dtype=np.float16)
    labels_train = labels_np[trainId]
    labels_val = labels_np[validId]

    print('Finish spliting dataset to train and test. ')


    adj_train = sp.csr_matrix(adj_train)
    adj_val = sp.csr_matrix(adj_val)

    graph_dgl = dgl.from_scipy(adj_train)
    graph_dgl.add_edges(graph_dgl.nodes(), graph_dgl.nodes())  # the library does not add self-loops
    num_nodes = graph_dgl.number_of_dst_nodes()
    adj_train = torch.tensor(adj_train.todense())  # use sparse man
    adj_train = adj_train + sp.eye(adj_train.shape[0]).todense()

    graph_dgl_val = dgl.from_scipy(adj_val)
    graph_dgl_val.add_edges(graph_dgl_val.nodes(), graph_dgl_val.nodes())  # the library does not add self-loops
    num_nodes_val = graph_dgl.number_of_dst_nodes()
    adj_val = torch.tensor(adj_val.todense())  # use sparse man
    adj_val = adj_val + sp.eye(adj_val.shape[0]).todense()

    if (type(feat_train) == np.ndarray):
        feat_train = torch.tensor(feat_train, dtype=torch.float32)
        feat_val = torch.tensor(feat_val, dtype=torch.float32)


    model = VGAE_FrameWork(num_of_comunities,
                            encoder = encoder_model,
                            decoder = decoder_model,
                            feature_decoder = feature_decoder,
                            feature_encoder = feature_encoder_model,
                            classifier=class_decoder)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    pos_wight = torch.true_divide((adj_train.shape[0] ** 2 - torch.sum(adj_train)), torch.sum(
        adj_train))  # addrressing imbalance data problem: ratio between positve to negative instance
    norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                             ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))
    pos_weight_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train)),
                                        torch.sum(feat_train))

    norm_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1]),
                                  (2 * (feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train))))

    pos_weight_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val)),
                                            torch.sum(feat_val))
    norm_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1]),
                                      (2 * (feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val))))

    lambda_1, lambda_2, lambda_3 = 1,1,1

    # logreg = LogisticRegression(solver='liblinear')
    # c = 2.0 ** np.arange(-10, 10)
    #
    # clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
    #                    param_grid=dict(estimator__C=c), n_jobs=4, cv=5,
    #                    verbose=0)

    w1 = torch.nn.Parameter(torch.tensor(0.5))
    w2 = torch.nn.Parameter(torch.tensor(0.5))
    w3 = torch.nn.Parameter(torch.tensor(0.5))
    optimizer_hyperparams = torch.optim.SGD([w1, w2, w3], lr=0.01)

    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all train nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train,
                                                                                targets, sampling_method,
                                                                                is_prior, train=True)
        # clf.fit(z.detach().numpy(), labels_train)
        # re_labels = clf.predict_proba(z.detach().numpy())
        # re_labels = torch.from_numpy(re_labels)
        # compute loss and accuracy
        z_kl, reconstruction_loss,posterior_cost_edges ,posterior_cost_features , posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat = optimizer_VAE(lambda_1, lambda_2,
                                                                                                lambda_3, labels_train,
                                                                                                re_labels, loss_type,
                                                                                                reconstructed_adj,
                                                                                                reconstructed_feat,
                                                                                                adj_train,
                                                                                                feat_train, norm_feat,
                                                                                                pos_weight_feat,
                                                                                                std_z, m_z, num_nodes,
                                                                                                pos_wight, norm)


        loss = reconstruction_loss + z_kl


        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print some metrics
        print("Epoch: {:03d} | Loss: {:05f} | edge_loss: {:05f} |feat_loss: {:05f} |node_classification_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
            epoch + 1, loss.item(), reconstruction_loss.item(),posterior_cost_edges.item() ,posterior_cost_features.item() , posterior_cost_classes.item(), z_kl.item(), acc))
    model.eval()

    # logreg = LogisticRegression(solver='liblinear')
    # c = 2.0 ** np.arange(-10, 10)
    #
    # clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
    #                    param_grid=dict(estimator__C=c), n_jobs=4, cv=5,
    #                    verbose=0)
    #
    # clf.fit(z.detach().numpy(), labels_train)
    return model, z

# def train_model():
#     val_loss = 0
#
#     return val_loss
