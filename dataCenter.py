#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:48:38 2023

@author: pnaddaf
"""

import sys
import os

from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import torch
from sklearn.preprocessing import OneHotEncoder
import json
import pickle
import zipfile

import dgl
from dgl.data import PubmedGraphDataset, CoauthorCSDataset

from networkx.readwrite import json_graph
from torch.hub import download_url_to_file

import copy


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


class DataCenter():
    """docstring for DataCenter"""

    def __init__(self):
        super().__init__()

    def load_dataSet(self, dataSet):
        if dataSet == 'photos' or dataSet == 'computers':
            labels = np.load("./datasets/" + dataSet + "/labels.npy")
            features = np.load("./datasets/" + dataSet + "/x.npy")

            # emb = torch.load("./datasets/"+dataSet+"/"+dataSet+"_x_gigamae.pt", map_location=torch.device('cpu'))
            # emb = emb.cpu().detach().numpy()
            # features = np.concatenate((features, emb), axis=1)

            adj = np.load("./datasets/" + dataSet + "/adj.npy")

            labels = np.asarray(labels, dtype=np.int64)
            test_indexs, val_indexs, train_indexs = self._split_data(labels, adj)
            encoder = OneHotEncoder(sparse_output=False)
            numerical_classes = labels.reshape(-1, 1)
            labels = encoder.fit_transform(numerical_classes)

            # test_indexs, val_indexs, train_indexs = self._split_data(labels)

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', features)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj)

        if dataSet == 'CiteSeer':
            names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
            objects = []
            for i in range(len(names)):
                with open("./datasets/citeseer/ind.{}.{}".format(dataSet, names[i]), 'rb') as f:
                    if sys.version_info > (3, 0):
                        objects.append(pkl.load(f, encoding='latin1'))
                    else:
                        objects.append(pkl.load(f))

            x, y, tx, ty, allx, ally, graph = tuple(objects)
            test_idx_reorder = parse_index_file("./datasets/citeseer/ind.{}.test.index".format(dataSet))
            test_idx_range = np.sort(test_idx_reorder)

            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            # features = np.asarray(features)
            # emb = torch.load("./datasets/citeseer/CiteSeer_x_gigamae.pt", map_location=torch.device('cpu'))
            # emb = emb.cpu().detach().numpy()
            # features = np.concatenate((features.toarray(), emb), axis=1)

            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

            labels = np.vstack((ally, ty))
            labels[test_idx_reorder, :] = labels[test_idx_range, :]

            labels = np.asarray(torch.argmax(torch.from_numpy(labels), dim=1), dtype=np.int64)
            test_indexs, val_indexs, train_indexs = self._split_data(labels, adj.toarray().astype(np.float32))
            encoder = OneHotEncoder(sparse_output= False)
            numerical_classes = labels.reshape(-1, 1)
            labels = encoder.fit_transform(numerical_classes)

            # test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', features.toarray())
            # setattr(self, dataSet + '_feats', features)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj.toarray().astype(np.float32))

        if dataSet == 'ppi':
            adj = np.load("./datasets/PPI/A.npy")
            features = np.load("./datasets/PPI/X.npy")

            test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', features)
            # setattr(self, dataSet+'_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj)

        if dataSet == "cs":
            graph = CoauthorCSDataset()[0]
            features = graph.ndata['feat'].numpy()
            # Get the edges
            src_nodes, dest_nodes = graph.edges()
            data = np.ones(len(dest_nodes))
            num_nodes = graph.number_of_nodes()
            adj = sp.coo_matrix((data, (src_nodes, dest_nodes)), shape=(num_nodes, num_nodes)).toarray()
            node_label = graph.ndata['label'].numpy()
            test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)
            setattr(self, dataSet + '_labels', node_label)
            setattr(self, dataSet + '_feats', features)
            setattr(self, dataSet + '_adj_lists', adj)

        if dataSet == 'Cora':
            cora_content_file = './datasets/Cora/cora.content'
            cora_cite_file = './datasets/Cora/cora.cites'

            with open(cora_content_file) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            id_list = []
            for x in content:
                x = x.split()
                id_list.append(int(x[0]))
            id_list = list(set(id_list))
            old_to_new_dict = {}
            for idd in id_list:
                old_to_new_dict[idd] = len(old_to_new_dict.keys())

            with open(cora_cite_file) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            edge_list = []
            for x in content:
                x = x.split()
                edge_list.append([old_to_new_dict[int(x[0])], old_to_new_dict[int(x[1])]])

            all_nodes = set()
            for pair in edge_list:
                all_nodes.add(pair[0])
                all_nodes.add(pair[1])

            adjancy_matrix = lil_matrix((len(all_nodes), len(all_nodes)))

            for pair in edge_list:
                adjancy_matrix[pair[0], pair[1]] = 1
                adjancy_matrix[pair[1], pair[0]] = 1

            feat_data = []
            labels = []  # label sequence of node
            node_map = {}  # map node to Node_ID
            label_map = {}  # map label to Label_ID
            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])
                    node_map[info[0]] = i
                    if not info[-1] in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            # emb = torch.load("./datasets/Cora/Cora_x_gigamae.pt", map_location=torch.device('cpu'))
            # emb = emb.cpu().detach().numpy()
            # feat_data = np.concatenate((feat_data, emb), axis=1)
            labels = np.asarray(labels, dtype=np.int64)
            test_indexs, val_indexs, train_indexs = self._split_data(labels, adjancy_matrix.toarray())
            encoder = OneHotEncoder(sparse_output= False)
            numerical_classes = labels.reshape(-1, 1)
            labels = encoder.fit_transform(numerical_classes)

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feat_data)
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adjancy_matrix.toarray())

        if dataSet == "IMDB":
            obj = []

            adj_file_name = "./datasets/IMDB/edges.pkl"

            with open(adj_file_name, 'rb') as f:
                obj.append(pkl.load(f))

            # merging diffrent edge type into a single adj matrix
            adj = lil_matrix(obj[0][0].shape)
            for matrix in obj[0]:
                adj += matrix

            matrix = obj[0]
            edge_labels = matrix[0] + matrix[1]
            edge_labels += (matrix[2] + matrix[3]) * 2

            node_label = []
            in_1 = matrix[0].indices.min()
            in_2 = matrix[0].indices.max() + 1
            in_3 = matrix[2].indices.max() + 1
            node_label.extend([0 for i in range(in_1)])
            node_label.extend([1 for i in range(in_1, in_2)])
            node_label.extend([2 for i in range(in_2, in_3)])

            obj = []
            with open("./datasets/IMDB/node_features.pkl", 'rb') as f:
                obj.append(pkl.load(f))
            feature = sp.csr_matrix(obj[0])
            feature = sp.csr_matrix(obj[0])

            index = -1
            labels = np.asarray(node_label, dtype=np.int64)
            test_indexs, val_indexs, train_indexs = self._split_data(labels[:index], adj)
            encoder = OneHotEncoder(sparse_output=False)
            numerical_classes = labels.reshape(-1, 1)
            labels = encoder.fit_transform(numerical_classes)

            # index = -1
            # test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feature[:index].toarray())
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj[:index, :index].toarray())
            setattr(self, dataSet + '_edge_labels', edge_labels[:index].toarray())

        if dataSet == "ACM":
            obj = []
            adj_file_name = "./datasets/ACM/edges.pkl"
            with open(adj_file_name, 'rb') as f:
                obj.append(pkl.load(f))

            adj = sp.csr_matrix(obj[0][0].shape)
            for matrix in obj:
                nnz = matrix[0].nonzero()  # indices of nonzero values
                for i, j in zip(nnz[0], nnz[1]):
                    adj[i, j] = 1
                    adj[j, i] = 1
                # adj +=matrix[0]

            # to fix the bug on running GraphSAGE
            adj = adj.toarray()
            for i in range(len(adj)):
                if sum(adj[i, :]) == 0:
                    idx = np.random.randint(0, len(adj))
                    adj[i, idx] = 1
                    adj[idx, i] = 1

            edge_labels = matrix[0] + matrix[1]
            edge_labels += (matrix[2] + matrix[3]) * 2

            node_label = []
            in_1 = matrix[0].indices.min()
            in_2 = matrix[0].indices.max() + 1
            in_3 = matrix[2].indices.max() + 1
            node_label.extend([0 for i in range(in_1)])
            node_label.extend([1 for i in range(in_1, in_2)])
            node_label.extend([2 for i in range(in_2, in_3)])

            obj = []
            with open("./datasets/ACM/node_features.pkl", 'rb') as f:
                obj.append(pkl.load(f))
            feature = sp.csr_matrix(obj[0])

            index = -1
            # test_indexs, val_indexs, train_indexs = self._split_data(feature[:index],adj[:index, :index])

            labels = np.asarray(node_label, dtype=np.int64)
            test_indexs, val_indexs, train_indexs = self._split_data(labels[:index], adj)
            encoder = OneHotEncoder(sparse_output=False)
            numerical_classes = labels.reshape(-1, 1)
            labels = encoder.fit_transform(numerical_classes)

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feature[:index].toarray())
            setattr(self, dataSet + '_labels', labels)
            setattr(self, dataSet + '_adj_lists', adj[:index, :index])
            setattr(self, dataSet + '_edge_labels', edge_labels[:index, :index].toarray())

        if dataSet == "pubmed":
            dataset = PubmedGraphDataset()
            graph = dataset[0]
            features = graph.ndata['feat'].numpy()
            # Get the edges
            src_nodes, dest_nodes = graph.edges()
            data = np.ones(len(dest_nodes))
            num_nodes = graph.number_of_nodes()
            adj = sp.coo_matrix((data, (src_nodes, dest_nodes)), shape=(num_nodes, num_nodes)).toarray()
            node_label = graph.ndata['label'].numpy()
            test_indexs, val_indexs, train_indexs = self._split_data(features.shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)
            setattr(self, dataSet + '_labels', node_label)
            setattr(self, dataSet + '_feats', features)
            setattr(self, dataSet + '_adj_lists', adj)



        elif dataSet == "DBLP":

            obj = []

            adj_file_name = "./datasets/DBLP/edges.pkl"

            with open(adj_file_name, 'rb') as f:
                obj.append(pkl.load(f))

            # merging diffrent edge type into a single adj matrix
            adj = sp.csr_matrix(obj[0][0].shape)
            for matrix in obj[0]:
                adj += matrix

            matrix = obj[0]
            edge_labels = matrix[0] + matrix[1]
            edge_labels += (matrix[2] + matrix[3]) * 2

            node_label = []
            in_1 = matrix[0].nonzero()[0].min()
            in_2 = matrix[0].nonzero()[0].max() + 1
            in_3 = matrix[3].nonzero()[0].max() + 1
            matrix[0].nonzero()
            node_label.extend([0 for i in range(in_1)])
            node_label.extend([1 for i in range(in_1, in_2)])
            node_label.extend([2 for i in range(in_2, in_3)])

            obj = []
            with open("./datasets/node_features.pkl", 'rb') as f:
                obj.append(pkl.load(f))
            feature = sp.csr_matrix(obj[0])

            index = -1000
            test_indexs, val_indexs, train_indexs = self._split_data(feature[:index].shape[0])

            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

            setattr(self, dataSet + '_feats', feature[:index].toarray())
            setattr(self, dataSet + '_labels', np.array(node_label[:index]))
            setattr(self, dataSet + '_adj_lists', adj[:index, :index].toarray())
            setattr(self, dataSet + '_edge_labels', edge_labels[:index].toarray())

    def _split_data(self, labels, adj, test_split=0.2, val_split=0.1):
        np.random.seed(123)
        num_nodes = labels.shape[0]
        num_classes = len(np.unique(labels))
        node_degree = np.sum(adj, axis=0)
        max_nodes = (int)(num_nodes / 20)
        train_nodes = np.argpartition(node_degree, -max_nodes)[-max_nodes:]

        nodes_dict = {}
        # create dict for each class
        for i in range(0, num_nodes):
            # if not i in train_nodes:
            nodes = nodes_dict.get(labels[i], [])
            nodes.append(i)
            #  nodes = np.random.permutation(nodes)
            nodes_dict[labels[i]] = nodes

        for i in range(0, num_classes):
            nodes = nodes_dict[i]
            nodes = np.random.permutation(nodes)
            nodes_dict[i] = nodes

        test_indexs = np.array([])
        val_indexs = np.array([])
        train_indexs = np.array([])

        for i in range(0, num_classes):
            n_nodes = len(nodes_dict[i])
            number_test = int(n_nodes * test_split)
            number_val = int(n_nodes * val_split)

            test_indexs = np.concatenate((test_indexs, nodes_dict[i][:number_test]))
            val_indexs = np.concatenate((val_indexs, nodes_dict[i][number_test:(number_test + number_val)]))
            train_indexs = np.concatenate((train_indexs, nodes_dict[i][(number_test + number_val):]))

        test_indexs = test_indexs.astype('int32')
        val_indexs = val_indexs.astype('int32')
        train_indexs = train_indexs.astype('int32')

        # nodes_dict_val = {}
        # nodes_dict_te = {}
        # nodes_dict_tr = {}
        # for i in range(0, len(test_indexs)):
        #     nodes = nodes_dict_te.get(labels[test_indexs[i]], [])
        #     nodes.append(i)
        #     nodes_dict_te[labels[test_indexs[i]]] = nodes
        #
        #
        # for i in range(0, len(val_indexs)):
        #     nodes = nodes_dict_val.get(labels[val_indexs[i]], [])
        #     nodes.append(i)
        #     nodes_dict_val[labels[val_indexs[i]]] = nodes
        #
        # for i in range(0, len(train_indexs)):
        #     nodes = nodes_dict_tr.get(labels[train_indexs[i]], [])
        #     nodes.append(i)
        #     nodes_dict_tr[labels[train_indexs[i]]] = nodes
        #
        # for i in range(0, num_classes):
        #     print(i)
        #     print(len(nodes_dict_tr[i]))
        #     print(len(nodes_dict_te[i]))
        #     print(len(nodes_dict_val[i]))

        return test_indexs, val_indexs, train_indexs


        # np.random.seed(123)
        # num_nodes = labels.shape[0]
        # num_classes = len(np.unique(labels))
        # node_degree = np.sum(adj, axis=0)
        # max_nodes = (int)(num_nodes / 20)
        # rand_indices = np.random.permutation(num_nodes)
        #
        # test_size = int(num_nodes * test_split)
        # val_size = int(num_nodes * val_split)
        # train_size = num_nodes - (test_size + val_size)
        #
        # test_indexs = rand_indices[:test_size]
        # val_indexs = rand_indices[test_size:(test_size + val_size)]
        # train_indexs = rand_indices[(test_size + val_size):]
        #
        # return test_indexs, val_indexs, train_indexs
        # np.random.seed(123)
        # num_nodes = labels.shape[0]
        # num_classes = len(np.unique(labels))
        #
        # nodes_dict = {}
        # # create dict for each class
        # for i in range(0, num_nodes):
        #     # if not i in train_nodes:
        #     nodes = nodes_dict.get(labels[i], [])
        #     nodes.append(i)
        #     #  nodes = np.random.permutation(nodes)
        #     nodes_dict[labels[i]] = nodes
        #
        # for i in range(0, num_classes):
        #     nodes = nodes_dict[i]
        #     nodes = np.random.permutation(nodes)
        #     nodes_dict[i] = nodes
        #
        # test_indexs = np.array([])
        # val_indexs = np.array([])
        # train_indexs = np.array([])
        #
        # len_test = int(num_nodes * test_split)
        # len_val = int(num_nodes * val_split)
        # len_train = num_nodes - (len_val + len_test)
        #
        # node_per_class = int(len_train / num_classes)
        # for i in range(0, num_classes):
        #     train_indexs = np.concatenate((train_indexs, nodes_dict[i][:node_per_class]))
        #
        # list_of_nodes = [x for x in range(0, num_nodes)]
        # available_nodes = [x for x in list_of_nodes if x not in train_indexs]
        #
        # test_indexs = available_nodes[:len_test]
        # val_indexs = available_nodes[len_test:len_test + len_val]
        # train_indexs = np.concatenate((train_indexs, available_nodes[len_test + len_val:]))
        #
        # return test_indexs, val_indexs, train_indexs.astype('i')

        # np.random.seed(123)
        # num_nodes = labels.shape[0]
        # num_classes = len(np.unique(labels))
        #
        # nodes_dict = {}
        # # create dict for each class
        # for i in range(0, num_nodes):
        #     # if not i in train_nodes:
        #     nodes = nodes_dict.get(labels[i], [])
        #     nodes.append(i)
        #     #  nodes = np.random.permutation(nodes)
        #     nodes_dict[labels[i]] = nodes
        #
        # for i in range(0, num_classes):
        #     nodes = nodes_dict[i]
        #     nodes = np.random.permutation(nodes)
        #     nodes_dict[i] = nodes
        #
        # len_min_class = len(nodes_dict[0])
        # for i in range(0, num_classes):
        #     if len(nodes_dict[i])<len_min_class:
        #         len_min_class = len(nodes_dict[i])
        #
        #
        # test_indexs = np.array([])
        # val_indexs = np.array([])
        # train_indexs = np.array([])
        #
        # for i in range(0, num_classes):
        #     n_nodes = len_min_class
        #     number_test = int(n_nodes * test_split)
        #     number_val = int(n_nodes * val_split)
        #     number_train = n_nodes-(number_test+number_val)
        #
        #     test_indexs = np.concatenate((test_indexs, nodes_dict[i][:number_test]))
        #     val_indexs = np.concatenate((val_indexs, nodes_dict[i][number_test:(number_test + number_val)]))
        #     train_indexs = np.concatenate((train_indexs, nodes_dict[i][(number_test + number_val):(number_test + number_val+number_train)]))
        #
        # train_indexs = train_indexs.astype('i')
        # val_indexs = val_indexs.astype('i')
        # test_indexs = test_indexs.astype('i')
        # node_d_te = {}
        # node_d_tr = {}
        # node_d_v = {}
        #
        # for i in test_indexs:
        #     nodes = node_d_te.get(labels[i], [])
        #     nodes.append(i)
        #     node_d_te[labels[i]] = nodes
        #
        # for i in val_indexs:
        #     nodes = node_d_v.get(labels[i], [])
        #     nodes.append(i)
        #     node_d_v[labels[i]] = nodes
        #
        # for i in train_indexs:
        #     nodes = node_d_tr.get(labels[i], [])
        #     nodes.append(i)
        #     node_d_tr[labels[i]] = nodes
        #
        # for i in range(0, num_classes):
        #     print(i)
        #     print(len(node_d_tr[i]))
        #     print(len(node_d_te[i]))
        #     print(len(node_d_v[i]))
        #
        # return test_indexs, val_indexs, train_indexs