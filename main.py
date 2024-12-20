#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:58:09 2023

@author: pnaddaf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:51:23 2022

@author: pnaddaf
"""
import sys
import os
import argparse
import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
import pyhocon
import dgl
import csv
from dgl.nn.pytorch import GraphConv as GraphConv
import copy
from dataCenter import *
from utils import *
from models import *
import time
import helper_opt as helper
import statistics
import warnings

warnings.simplefilter('ignore')

# %%  arg setup

##################################################################


parser = argparse.ArgumentParser(description='Inductive')

parser.add_argument('--e', type=int, dest="epoch_number", default=100, help="Number of Epochs")
parser.add_argument('--dataSet', type=str, default="Cora_dgl")
parser.add_argument('--loss_type', dest="loss_type", default="1", help="type of combination between loss_A and loss_F")
parser.add_argument('--sampling_method', dest="sampling_method", default="deterministic", help="This var shows sampling method it could be: monte, importance_sampling, deterministic")
parser.add_argument('--method', dest="method", default="single", help="This var shows method it could be: multi, single")
parser.add_argument('--iterative', dest="iterative", default="False", type=str, help="This flag is used if want to have iterative link prediction")
parser.add_argument('--tuning', dest="tuning", default="False", type=str, help="This flag is used if want to tune hyperparameters in helper_opt")


parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--num_node', dest="num_node", default=-1, type=str,
                    help="the size of subgraph which is sampled; -1 means use the whole graph")
parser.add_argument('--decoder_type', dest="decoder_type", default="ML_SBM",help="the decoder type")
parser.add_argument('--encoder_type', dest="encoder_type", default="Multi_GAT",
                    help="the encoder type, Either Multi_GIN, Multi_GCN, Multi_GAT, Multi_SAGE")
parser.add_argument('--NofRels', dest="num_of_relations", default=1,
                    help="Number of latent or known relation; number of deltas in SBM")
parser.add_argument('--NofCom', dest="num_of_comunities", default=128,
                    help="Number of comunites, tor latent space dimention; len(z)")
parser.add_argument('-BN', dest="batch_norm", default=True,
                    help="either use batch norm at decoder; only apply in multi relational decoders")
parser.add_argument('--DR', dest="DropOut_rate", default=.0, help="drop out rate")
parser.add_argument('--encoder_layers', dest="encoder_layers", default="64", type=str,
                    help="a list in which each element determine the size of gcn; Note: the last layer size is determine with -NofCom")
parser.add_argument('--lr', dest="lr", default=0.01, help="model learning rate")
parser.add_argument('--is_prior', dest="is_prior", default=False, help="This flag is used for sampling methods")
parser.add_argument('--targets', dest="targets", default=[], help="This list is used for sampling")
parser.add_argument('--fully_inductive', dest="fully_inductive", default=False,
                    help="This flag is used if want to have fully o semi inductive link prediction")
parser.add_argument('--transductive', dest="transductive", default="False", type=str,
                    help="This flag is used if want to have transductive link prediction")
parser.add_argument('--edge_base', dest="edge_base", default="True", type=str,
                    help="This flag is used if want to have edge base data splitting")

args = parser.parse_args()
fully_inductive = args.fully_inductive

print("")
print("SETING: " + str(args))

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    print('Using device', device_id, torch.cuda.get_device_name(device_id))
else:
    device_id = 'cpu'

device = torch.device(device_id)



random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# %% load data
ds = args.dataSet
dataCenter = DataCenter()
dataCenter.load_dataSet(ds)
org_adj = getattr(dataCenter, ds + '_adj_lists')
features = torch.FloatTensor(getattr(dataCenter, ds + '_feats'))
labels = torch.FloatTensor(getattr(dataCenter, ds + '_labels')).to(device)
val_indx = getattr(dataCenter, ds + '_val_edge_idx')
train_indx = getattr(dataCenter, ds + '_train_edge_idx')
ignore_edges = getattr(dataCenter, ds + '_ignore_edges_inx')
adj_test = getattr(dataCenter, ds + '_adj_test')





#  train inductive_model
inductive_model, z_p = helper.train_model(dataCenter, features.to(device),
                                         args, device)
# inductive_model, z_p = helper.train_PNModel(dataCenter, features.to(device),
#                                          args, device)


# Split A into test and train
trainId = getattr(dataCenter, ds + '_train')
testId = getattr(dataCenter, ds + '_test')


# testId = trainId



# defining metric lists
auc_list = []
acc_list = []
ap_list = []
precision_list = []
recall_list = []
HR_list = []
pr_auc_list = []

auc_list_label = []
acc_list_label = []
ap_list_label = []
precision_list_label = []
recall_list_label = []
F1_list_label = []

num_neighbour = []
wrong_pred = []


method = args.method
if method=='multi':
    single_link = False
    multi_link = True
    multi_single_link_bl = False
elif method == 'single':
    single_link = True
    multi_link = False
    multi_single_link_bl = False



pred_single_link = []
true_single_link = []
pred_single_label = []
true_single_label = []
pred_multi_label = []
true_multi_label = []
targets = []
target_ids = []
sampling_method = args.sampling_method

pred_multi = np.array([])
true_multi = np.array([])

if fully_inductive:
    res = org_adj.nonzero()
    index = np.where(np.isin(res[0], testId) & np.isin(res[1], trainId) | (
                np.isin(res[1], testId) & np.isin(res[0], trainId)))  # find edges that connect test to train
    i_list = res[0][index]
    j_list = res[1][index]
    org_adj[i_list, j_list] = 0  # set all the in between edges to 0

# run recognition separately for the case of single_link
std_z_recog, m_z_recog, z_recog, re_adj_recog, re_feat_recog, re_recog_labels = run_network(features, org_adj, labels, inductive_model, targets, sampling_method,
                                                            is_prior=False)
if args.edge_base:
    res = adj_test.nonzero()
    test_edges = np.array([res[0], res[1]]).T
    auc, val_acc, val_ap, precision, recall, HR, auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = test(test_edges,
                                                                                                            org_adj,
                                                                                                            run_network,
                                                                                                            features,
                                                                                                            labels,
                                                                                                            inductive_model,
                                                                                                            targets,
                                                                                                            sampling_method)


else:
    res = org_adj.nonzero()
    index = np.where(np.isin(res[0], testId))  # only one node of the 2 ends of an edge needs to be in testId
    idd_list = res[0][index]
    neighbour_list = res[1][index]
    sample_list = random.sample(range(0, len(idd_list)), 100)

    # run prior network separately
    correct_subgraph = 0
    counter = 0
    target_edges = []

    # auc, val_acc, val_ap, precision, recall, HR, auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = test(test_edges, org_adj, run_network, features, labels, inductive_model, targets, sampling_method)
    for i in sample_list:
        start_time = time.time()
        print(counter)
        counter+= 1
        targets = []
        idd = idd_list[i]
        neighbour_id = neighbour_list[i]
        adj_list_copy = copy.deepcopy(org_adj)
        neigbour_prob_single = 1

        if single_link:

            adj_list_copy = copy.deepcopy(org_adj)
            adj_list_copy[idd, neighbour_id] = 0  # find a test edge and set it to 0
            adj_list_copy[neighbour_id, idd] = 0  # find a test edge and set it to 0

            targets.append(idd)
            targets.append(neighbour_id)
            target_edges.append([idd, neighbour_id])
            target_edges.append([neighbour_id, idd])
            target_ids.append(idd)
            target_ids.append(neighbour_id)


            # run prior
            std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features,
                                                                                                        adj_list_copy,
                                                                                                        labels,
                                                                                                        inductive_model,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior=True)

            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            re_label_prior_sig = torch.sigmoid(re_prior_labels)
            pred_single_link.append(re_adj_prior_sig[idd, neighbour_id].tolist())
            true_single_link.append(org_adj[idd, neighbour_id].tolist())
            pred_single_label.append(re_label_prior_sig[idd])
            true_single_label.append(labels[idd])

        if multi_link:
            true_multi_links = org_adj[idd].nonzero()
            false_multi_links = np.array(random.sample(list(np.nonzero(org_adj[idd] == 0)[0]), len(true_multi_links[0])))

            target_list = [[idd, i] for i in list(true_multi_links[0])]
            target_list.extend([[idd, i] for i in list(false_multi_links)])
            target_list = np.array(target_list)


            targets = []
            targets.append(idd)


            # # run prior
            # if the selected method is monte, this would be (all 0 + MC) or (MC) and if the selected method is IS, this would be IS
            adj_list_copy = copy.deepcopy(org_adj)
            adj_list_copy[idd, :] = 0  # set all the neigbours to 0
            adj_list_copy[:, idd] = 0  # set all the neigbours to 0

            # run prior
            # target_edges.extend(target_list)
            if args.iterative == "True":
                c = 0
                for e in target_list:


                    std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features,
                                                                                                                adj_list_copy,
                                                                                                                labels,
                                                                                                                inductive_model,
                                                                                                                targets,
                                                                                                                sampling_method,
                                                                                                                is_prior=True)


                    re_adj_prior_sig = torch.sigmoid(re_adj_prior)

                    # _, _, _, _, _, _, th = get_metrics(target_list, org_adj, re_adj_prior_sig)
                    target_edges.append((e, re_adj_prior_sig[e[0]][e[1]]))
                    # if re_adj_prior_sig[e[0]][e[1]] > 0.723:
                    adj_list_copy[e[0]][e[1]] = 1
                    adj_list_copy[e[1]][e[0]] = 1
                    if adj_list_copy[e[1]][e[0]]==1 and org_adj[e[1]][e[0]]==0:
                        c += 1
                        # print(adj_list_copy[e[1]][e[0]], org_adj[e[1]][e[0]])
                wrong_pred.append(c/len(target_list))
                num_neighbour.append(len(target_list))

                for e, p in target_edges:
                    re_adj_prior_sig[e[0]][e[1]] = p
                    re_adj_prior_sig[e[1]][e[0]] = p

            else:
                std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features,
                                                                                                            adj_list_copy,
                                                                                                            labels,
                                                                                                            inductive_model,
                                                                                                            targets,
                                                                                                            sampling_method,
                                                                                                            is_prior=True)

                re_adj_prior_sig = torch.sigmoid(re_adj_prior)

            re_label_prior_sig = torch.sigmoid(re_prior_labels)
            pred_multi_label.append(re_label_prior_sig[idd])
            true_multi_label.append(labels[idd])


            auc, val_acc, val_ap, precision, recall, HR, pr_auc = get_metrics(target_list, org_adj, re_adj_prior_sig)


            auc_list.append(auc)
            acc_list.append(val_acc)
            ap_list.append(val_ap)
            precision_list.append(precision)
            recall_list.append(recall)
            HR_list.append(HR)
            # pr_auc_list.append(pr_auc)





    # consider negative edges for single link
    if single_link:
        false_count = len(pred_single_link)
        res = np.argwhere(org_adj == 0)
        np.random.shuffle(res)
        index = np.where(np.isin(res[:, 0], testId))  # only one node of the 2 ends of an edge needs to be in testId
        test_neg_edges = res[index]
        for test_neg_edge in test_neg_edges[:false_count]:
            targets = []
            idd = test_neg_edge[0]
            neighbour_id = test_neg_edge[1]
            adj_list_copy = copy.deepcopy(org_adj)
            adj_list_copy[idd, neighbour_id] = 0
            adj_list_copy[neighbour_id, idd] = 0
            targets.append(idd)
            targets.append(neighbour_id)
            target_edges.append([idd, neighbour_id])
            target_edges.append([neighbour_id, idd])
            target_ids.append(idd)
            target_ids.append(neighbour_id)

            # to update mq and sq for the case of importance_sampling

            std_z_recog, m_z_recog, z_recog, re_adj_recog, re_feat_recog, re_recog_labels = run_network(features,
                                                                                                        adj_list_copy,
                                                                                                        labels,
                                                                                                        inductive_model,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior=False)

            std_z_prior, m_z_prior, z_prior, re_adj_prior, re_feat_prior, re_prior_labels = run_network(features, org_adj,
                                                                                                        labels,
                                                                                                        inductive_model,
                                                                                                        targets,
                                                                                                        sampling_method,
                                                                                                        is_prior=True)



            re_adj_prior_sig = torch.sigmoid(re_adj_prior)
            re_label_prior_sig = torch.sigmoid(re_prior_labels)
            pred_single_link.extend([re_adj_prior_sig[idd, neighbour_id].tolist()])
            true_single_link.extend([org_adj[idd, neighbour_id].tolist()])
            pred_single_label.extend([re_label_prior_sig[idd]])
            true_single_label.extend([labels[idd]])

        auc, val_acc, val_ap, precision, recall, HR = roc_auc_single(pred_single_link, true_single_link)
        auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = roc_auc_estimator_labels(pred_single_label, true_single_label,labels)

        auc_list.append(auc)
        acc_list.append(val_acc)
        ap_list.append(val_ap)
        precision_list.append(precision)
        recall_list.append(recall)
        HR_list.append(HR)

        auc_list_label.append(auc_l)
        acc_list_label.append(acc_l)
        ap_list_label.append(ap_l)
        precision_list_label.append(precision_l)
        recall_list_label.append(recall_l)
        F1_list_label.append(F1_score)

    if multi_link:
        auc_l, acc_l, ap_l, precision_l, recall_l, F1_score = roc_auc_estimator_labels(pred_multi_label,
                                                                                       true_multi_label, labels)
        auc_list_label.append(auc_l)
        acc_list_label.append(acc_l)
        ap_list_label.append(ap_l)
        precision_list_label.append(precision_l)
        recall_list_label.append(recall_l)
        F1_list_label.append(F1_score)

auc_list.append(auc)
acc_list.append(val_acc)
ap_list.append(val_ap)
precision_list.append(precision)
recall_list.append(recall)
HR_list.append(HR)
auc_list_label.append(auc_l)
acc_list_label.append(acc_l)
ap_list_label.append(ap_l)
precision_list_label.append(precision_l)
recall_list_label.append(recall_l)
F1_list_label.append(F1_score)


# Print results
if fully_inductive:
    save_recons_adj_name =args.encoder_type[-3:] + "_" + args.sampling_method + "_fully_" + args.method + "_" + args.dataSet
else:
    save_recons_adj_name = args.encoder_type[-3:] + "_" + args.sampling_method + "_semi_" + args.method + "_" + args.dataSet
if args.transductive == "True":
    save_recons_adj_name = "Trans_"+ save_recons_adj_name
else:
    save_recons_adj_name = "Ind_" + save_recons_adj_name

save_recons_adj_name = save_recons_adj_name + "_" + args.loss_type
print(save_recons_adj_name)
end_time = time.time()
print("time:")
# print(end_time-start_time)
if args.iterative == "True":
    print("Link Prediction")
    print("auc= %.3f , acc= %.3f ap= %.3f , precision= %.3f , recall= %.3f , HR= %.3f, nn= %.3f, w_ratio= %.3f" % (
    statistics.mean(auc_list), statistics.mean(acc_list), statistics.mean(ap_list), statistics.mean(precision_list),
    statistics.mean(recall_list), statistics.mean(HR_list), statistics.mean(num_neighbour)))
    with open('./results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["itr_"+save_recons_adj_name, statistics.mean(auc_list), statistics.mean(acc_list),
             statistics.mean(ap_list),
             statistics.mean(precision_list), statistics.mean(recall_list), statistics.mean(HR_list), statistics.mean(num_neighbour)])
else:
    print("Link Prediction")
    print("auc= %.3f , acc= %.3f ap= %.3f , precision= %.3f , recall= %.3f , HR= %.3f" % (
    statistics.mean(auc_list), statistics.mean(acc_list), statistics.mean(ap_list), statistics.mean(precision_list),
    statistics.mean(recall_list), statistics.mean(HR_list)))
    with open('./results.csv', 'a', newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            [save_recons_adj_name, statistics.mean(auc_list), statistics.mean(acc_list),
             statistics.mean(ap_list),
             statistics.mean(precision_list), statistics.mean(recall_list), statistics.mean(HR_list)])

# print("Link Prediction std")
# print("auc= %.3f , acc= %.3f ap= %.3f , precision= %.3f , recall= %.3f , HR= %.3f, pr_auc=%3f" % (
# statistics.stdev(auc_list), statistics.stdev(acc_list), statistics.stdev(ap_list), statistics.stdev(precision_list),
# statistics.stdev(recall_list), statistics.stdev(HR_list), statistics.stdev(pr_auc_list)))



print("Node Classification")
print("auc= %.3f , acc= %.3f ap= %.3f , precision= %.3f , recall= %.3f , F1_Score= %.3f" % (
statistics.mean(auc_list_label), statistics.mean(acc_list_label), statistics.mean(ap_list_label),
statistics.mean(precision_list_label), statistics.mean(recall_list_label), statistics.mean(F1_list_label)))


with open('./results.csv', 'a', newline="\n") as f:
    writer = csv.writer(f)
    writer.writerow(["labels_" + save_recons_adj_name, statistics.mean(auc_list_label), statistics.mean(acc_list_label),
                     statistics.mean(ap_list_label), statistics.mean(precision_list_label),
                     statistics.mean(recall_list_label), statistics.mean(F1_list_label)])
