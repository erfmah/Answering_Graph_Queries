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
import csv
from bayes_opt import BayesianOptimization


# %% KDD model
def train_model(dataCenter, features, args, device):
    dataset = args.dataSet
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

    elif encoder == "Multi_SAGE":
        encoder_model = multi_layer_SAGE(num_of_comunities, latent_dim=num_of_comunities, layers=encoder_layers)

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
    pos_wight_val = torch.true_divide((adj_val.shape[0] ** 2 - torch.sum(adj_val)), torch.sum(
        adj_val))
    norm = torch.true_divide(adj_train.shape[0] * adj_train.shape[0],
                             ((adj_train.shape[0] * adj_train.shape[0] - torch.sum(adj_train)) * 2))
    norm_val = torch.true_divide(adj_val.shape[0] * adj_val.shape[0],
                             ((adj_val.shape[0] * adj_val.shape[0] - torch.sum(adj_val)) * 2))
    pos_weight_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train)),
                                        torch.sum(feat_train))

    norm_feat = torch.true_divide((feat_train.shape[0] * feat_train.shape[1]),
                                  (2 * (feat_train.shape[0] * feat_train.shape[1] - torch.sum(feat_train))))

    pos_weight_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val)),
                                            torch.sum(feat_val))
    norm_feat_val = torch.true_divide((feat_val.shape[0] * feat_val.shape[1]),
                                      (2 * (feat_val.shape[0] * feat_val.shape[1] - torch.sum(feat_val))))


    pbounds = {
        'lambda_1': (0.0, 1.0),
        'lambda_2': (0.0, 1.0),
        'lambda_3': (0.0, 1.0)
    }
    optimizer_function = make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train, adj_val, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer)
    optimizer_hp = BayesianOptimization(
        f=optimizer_function,
        pbounds=pbounds,
        random_state=42
    )
    optimizer_hp.maximize(
        init_points=10,
        n_iter=20
    )
    print(optimizer_hp.max)

    # Extract and print the best values for weight1 and weight2
    best_params = optimizer_hp.max['params']
    lambda_1= best_params['lambda_1']
    lambda_2= best_params['lambda_2']
    lambda_3 = best_params['lambda_3']

    print("weights:", lambda_1, lambda_2, lambda_3)
    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all train nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train,
                                                                                targets, sampling_method,
                                                                                is_prior, train=True)

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

    with open('./weights.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            [args.dataSet, lambda_1, lambda_2, lambda_3])

    return model, z


def tune_model(params, labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer):
    lambda_1, lambda_2, lambda_3 = params
    best_auc = 0
    # with open('./results_csv/best_auc.csv', newline='') as f:
    #     reader = csv.DictReader(f)
    #     for q in reader:
    #         best_auc = float(q['auc'])

    # best_validation_loss = 0
    print(lambda_1, lambda_2, lambda_3)


    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train,
                                                                                targets, sampling_method,
                                                                                is_prior, train=True)
        # compute loss and accuracy
        z_kl, reconstruction_loss,posterior_cost_edges ,posterior_cost_features , posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat = optimizer_VAE(lambda_1, lambda_2,
                                                                                                lambda_3, labels_train,
                                                                                                re_labels, loss_type,
                                                                                                reconstructed_adj,
                                                                                                reconstructed_feat,
                                                                                                adj_train_org,
                                                                                                feat_train, norm_feat,
                                                                                                pos_weight_feat,
                                                                                                std_z, m_z, num_nodes,
                                                                                                pos_wight, norm)
        loss = reconstruction_loss + z_kl

        # reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()

        model.eval()

        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some metrics
        print(
            "Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
                epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
    model.eval()
    with torch.no_grad():

        std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(graph_dgl_val,
                                                                                                        feat_val,
                                                                                                        labels_val,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior,
                                                                                                        train=True)
        z_kl_val, val_reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, val_acc, val_recons_loss, loss_adj_val, loss_feat_val = optimizer_VAE(lambda_1,
                                                                                                            lambda_2,
                                                                                                            lambda_3,
                                                                                                            labels_val,
                                                                                                            re_labels_val,
                                                                                                            loss_type,
                                                                                                            reconstructed_adj_val,
                                                                                                            reconstructed_feat_val,
                                                                                                            adj_val_org,
                                                                                                            feat_val,
                                                                                                            norm_feat_val,
                                                                                                            pos_weight_feat_val,
                                                                                                            std_z_val,
                                                                                                            m_z_val,
                                                                                                            num_nodes_val,
                                                                                                            pos_wight_val,
                                                                                                            norm_val)
        val_loss_total = val_reconstruction_loss + z_kl_val
        # with open('./results_csv/loss_val.csv', 'a') as f:
        #     wtr = csv.writer(f)
        #     wtr.writerow([loss.item()])

    # y_true_feat = (torch.flatten(feat_val)).cpu().detach().numpy()
    # y_pred_feat = (torch.flatten(torch.sigmoid(reconstructed_feat_val))).cpu().detach().numpy()
    # index_sample_0_feat = np.random.choice(np.where(y_true_feat == 1)[0], 100)
    # index_sample_1_feat = np.random.choice(np.where(y_true_feat == 0)[0], 100)
    # index_sample_feat = np.concatenate((index_sample_0_feat, index_sample_1_feat))
    # auc_feat = roc_auc_score(y_score=y_pred_feat[index_sample_feat], y_true=y_true_feat[index_sample_feat])

    # y_true_adj = (torch.flatten(adj_val_org)).cpu().detach().numpy()
    # y_pred_adj = (torch.flatten(torch.sigmoid(reconstructed_adj_val))).cpu().detach().numpy()
    # index_sample_0_adj = np.random.choice(np.where(y_true_adj == 1)[0], 100)
    # index_sample_1_adj = np.random.choice(np.where(y_true_adj == 0)[0], 100)
    # index_sample_adj = np.concatenate((index_sample_0_adj, index_sample_1_adj))
    # auc_adj = roc_auc_score(y_score=y_pred_adj[index_sample_adj], y_true=y_true_adj[index_sample_adj])
    #
    # auc_labels = roc_auc_score(y_score= re_labels_val, y_true= labels_val)
    #
    # auc_val = -1*(auc_feat+auc_adj+auc_labels)
    #
    # if best_auc > auc_val:
    #     best_auc = auc_val
    #     torch.save(model.state_dict(), 'best_model_' + dataset + '.pt')
    #     with open('./results_csv/best_auc.csv', 'a') as f:
    #         wtr = csv.writer(f)
    #         wtr.writerow([best_auc])
    #
    #
    #
    # return auc_val

    w_l = weight_labels(labels_val)
    posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(reconstructed_adj_val, adj_val_org,
                                                                     pos_weight=pos_wight_val)
    posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat_val, feat_val,
                                                                             pos_weight=pos_weight_feat)
    posterior_cost_classes = F.cross_entropy(re_labels_val, (torch.tensor(labels_val).to(torch.float64)), weight=w_l)

    cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes

    return cost.item()

# z_kl, reconstruction_loss, acc, val_recons_loss = optimizer_VAE_em(alpha, mask_index, not_masked_index, reconstructed_adj, reconstructed_feat,
#                                                                adj_train_org, feat_train, norm_feat,pos_weight_feat,
#                                                                std_z, m_z, num_nodes, pos_wight_masked, pos_wight_not_masked, norm_masked, norm_not_masked )


# train_auc, train_acc, train_ap, train_conf = roc_auc_estimator_train(train_true, train_false,
#                                                       reconstructed_adj, adj_train)

# if split_the_data_to_train_test == True:
#     std_z, m_z, z, reconstructed_adj_val = model(graph_dgl_val, feat_val, is_prior, train=False)
#     reconstructed_adj_val = torch.sigmoid(reconstructed_adj_val).detach().numpy()
#     val_auc, val_acc, val_ap, val_conf = roc_auc_estimator_train(val_true, val_false,
#                                                     reconstructed_adj_val, adj_val)

#     # keep the history to plot
#     pltr.add_values(epoch, [loss.item(), train_acc,  reconstruction_loss.item(), z_kl, train_auc],
#                     [None, val_acc, val_recons_loss,None, val_auc  # , val_ap
#                         ], redraw=False)  # ["Accuracy", "Loss", "AUC", "AP"]
# else:
#     # keep the history to plot
#     pltr.add_values(epoch, [acc, loss.item(), None  # , None
#                             ],
#                     [None, None, None  # , None
#                       ], redraw=False)  # ["Accuracy", "loss", "AUC", "AP"])

# # Ploting the recinstructed Graph
# if epoch % visulizer_step == 0:
#     # pltr.redraw()
#     print("Val conf:", )
#     print(val_conf, )
#     print("Train Conf:")
#     print(train_conf)
def optimize_weights(lambda_1, lambda_2, lambda_3,labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer):
    for epoch in range(epoch_number):
        model.train()
        # forward propagation by using all nodes
        std_z, m_z, z, reconstructed_adj, reconstructed_feat, re_labels = model(graph_dgl, feat_train, labels_train,
                                                                                targets, sampling_method,
                                                                                is_prior, train=True)
        # compute loss and accuracy
        z_kl, reconstruction_loss, posterior_cost_edges, posterior_cost_features, posterior_cost_classes, acc, val_recons_loss, loss_adj, loss_feat = optimizer_VAE(
            lambda_1, lambda_2,
            lambda_3, labels_train,
            re_labels, loss_type,
            reconstructed_adj,
            reconstructed_feat,
            adj_train_org,
            feat_train, norm_feat,
            pos_weight_feat,
            std_z, m_z, num_nodes,
            pos_wight, norm)
        loss = reconstruction_loss + z_kl

        # reconstructed_adj = torch.sigmoid(reconstructed_adj).detach().numpy()

        model.eval()

        model.train()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print some metrics
        print(
            "Epoch: {:03d} | Loss: {:05f} | Reconstruction_loss: {:05f} | z_kl_loss: {:05f} | Accuracy: {:03f}".format(
                epoch + 1, loss.item(), reconstruction_loss.item(), z_kl.item(), acc))
    model.eval()
    with torch.no_grad():
        std_z_val, m_z_val, z_val, reconstructed_adj_val, reconstructed_feat_val, re_labels_val = model(graph_dgl_val,
                                                                                                        feat_val,
                                                                                                        labels_val,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior,
                                                                                                        train=True)

    w_l = weight_labels(labels_val)
    posterior_cost_edges = norm * F.binary_cross_entropy_with_logits(reconstructed_adj_val, adj_val_org,
                                                                     pos_weight=pos_wight_val)
    posterior_cost_features = norm_feat * F.binary_cross_entropy_with_logits(reconstructed_feat_val, feat_val,
                                                                             pos_weight=pos_weight_feat)
    posterior_cost_classes = F.cross_entropy(re_labels_val, (torch.tensor(labels_val).to(torch.float64)), weight=w_l)

    cost = posterior_cost_edges + posterior_cost_features + posterior_cost_classes

    return -1*cost.item()

# def optimize_weights_wrapper(weight1, weight2):
#     return optimize_weights(lambda_1, lambda_2, lambda_3,labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
#                 feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
#                 pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
#                 pos_wight_val, norm_val, optimizer)

def make_optimizer_wrapper(labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer):
    def optimize_weights_wrapper(lambda_1, lambda_2, lambda_3):
        return optimize_weights(lambda_1, lambda_2, lambda_3,labels_train, labels_val, dataset, epoch_number, model, graph_dgl, graph_dgl_val, feat_train,
                feat_val, targets, sampling_method, is_prior, loss_type, adj_train_org, adj_val_org, norm_feat,
                pos_weight_feat, norm_feat_val, pos_weight_feat_val, num_nodes, num_nodes_val, pos_wight, norm,
                pos_wight_val, norm_val, optimizer)
    return optimize_weights_wrapper