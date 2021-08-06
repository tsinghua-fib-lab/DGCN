#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import dgl.function as fn


class NodeUpdate(nn.Module):

    def __init__(self, in_feats, out_feats):

        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = nn.Tanh()

    def forward(self, node):

        h = node.data['h']
        h = self.linear(h)

        h = self.activation(h)

        return {'activation': h}


class DGLGCN(nn.Module):

    def __init__(self, flags_obj, dm):

        super(DGLGCN, self).__init__()

        self.num_users = dm.n_user
        self.num_items = dm.n_item
        self.num_nodes = self.num_users + self.num_items
        self.embeddings = Parameter(torch.FloatTensor(self.num_nodes, flags_obj.embedding_size))

        self.hop = flags_obj.hop

        self.layers = nn.ModuleList()
        for _ in range(self.hop):
            self.layers.append(NodeUpdate(flags_obj.embedding_size, flags_obj.embedding_size))

        self.dropout = flags_obj.dropout

        self.init_params()

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings.size(1))
        self.embeddings.data.uniform_(-stdv, stdv)

    def encode(self, data_flows, training=True):

        x = self.embeddings

        nf = next(iter(data_flows))
        nf.copy_from_parent()

        nf.layers[0].data['activation'] = x[nf.layers[0].data['feature']]

        for i, layer in enumerate(self.layers):

            h = nf.layers[i].data.pop('activation')
            h = F.dropout(h, p=self.dropout, training=training)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')

        return h

    def forward(self, data_flows, node_index, user_index, item_index):

        h = self.encode(data_flows)

        user = h[user_index]
        item = h[item_index]

        score = torch.sum(user * item, 2)

        return score

    def get_embeddings(self, node_flow):

        h = self.encode(node_flow, False)

        user_embeddings = h[:self.num_users].detach().cpu().numpy().astype('float32')
        item_embeddings = h[self.num_users:].detach().cpu().numpy().astype('float32')

        return user_embeddings, item_embeddings


class DivDGLGCN(nn.Module):

    def __init__(self, flags_obj, dm):

        super(DivDGLGCN, self).__init__()

        self.encoder = DGLGCN(flags_obj, dm)
        self.feature_classifier = nn.Linear(flags_obj.embedding_size, dm.n_feature)

        self.init_params()

    def init_params(self):

        self.encoder.init_params()

    def forward(self, data_flows, node_index, user_index, item_index):

        h = self.encoder.encode(data_flows)

        user = h[user_index]
        item = h[item_index]

        score_interaction = torch.sum(user * item, 2)

        item_feature = item.clone()
        item_feature.register_hook(lambda grad: -grad)

        score_feature = self.feature_classifier(item_feature)

        return score_interaction, score_feature

    def get_embeddings(self, node_flow):

        return self.encoder.get_embeddings(node_flow)
