#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import torch
import torch.optim as optim


import data
import model
import utils
import candidate_generator as cg
import config.const as const_util

import os


class Recommender(object):

    def __init__(self, flags_obj, workspace, dm):

        self.dm = dm
        self.model_name = flags_obj.model
        self.flags_obj = flags_obj
        self.set_device()
        self.set_model()
        self.workspace = workspace
    
    def set_device(self):

        self.device  = utils.ContextManager.set_device(self.flags_obj)
    
    def set_model(self):

        raise NotImplementedError
    
    def transfer_model(self):

        self.model = self.model.to(self.device)
    
    def save_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        torch.save(self.model.state_dict(), model_path)
    
    def load_ckpt(self, epoch):

        ckpt_path = os.path.join(self.workspace, const_util.ckpt)
        model_path = os.path.join(ckpt_path, 'epoch_' + str(epoch) + '.pth')
        self.model.load_state_dict(torch.load(model_path, map_location='cuda:{}'.format(self.flags_obj.gpu_id)))
    
    def get_dataloader(self):

        raise NotImplementedError

    def get_optimizer(self):

        raise NotImplementedError
    
    def inference(self, sample):

        raise NotImplementedError
    
    def make_cg(self):

        raise NotImplementedError
    
    def cg(self, users, topk):

        raise NotImplementedError


class GCNRecommender(Recommender):

    def __init__(self, flags_obj, workspace, dm):

        super(GCNRecommender, self).__init__(flags_obj, workspace, dm)
        self.set_gm()

    def set_gm(self):

        self.gm = utils.DGLGraphManager(self.flags_obj, self.dm)

    def set_model(self):

        self.model = model.DGLGCN(self.flags_obj, self.dm)

    def get_dataloader(self):

        return data.FactorizationDataProcessor.get_point_dataloader(self.flags_obj, self.dm)

    def get_optimizer(self):

        return optim.Adam(self.model.parameters(), lr=self.flags_obj.lr, weight_decay=self.flags_obj.weight_decay, betas=(0.5, 0.99), amsgrad=True)

    def inference(self, sample):

        user, item, label = sample
        item = item + self.dm.n_user

        data_flows, node_index, user_index, item_index = self.gm.point_sample_neighbors(user, item)

        user = user.to(self.device)
        item = item.to(self.device)
        label = label.to(self.device)

        score = self.model(data_flows, node_index, user_index, item_index)

        return score, label

    def make_cg(self):

        user_embeddings, item_embeddings = self.get_embeddings()

        self.generator = cg.FaissInnerProductMaximumSearchGenerator(self.flags_obj, item_embeddings)
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def get_embeddings(self):

        node_flow = self.gm.get_complete_node_flows()
        user_embeddings, item_embeddings = self.model.get_embeddings(node_flow)

        return user_embeddings, item_embeddings

    def cg(self, users, topk):

        return self.generator.generate(self.user_embeddings[users], topk)


class DGCNRecommender(GCNRecommender):

    def __init__(self, flags_obj, workspace, dm):

        super(DGCNRecommender, self).__init__(flags_obj, workspace, dm)

    def get_dataloader(self):

        return data.FactorizationDataProcessor.get_adv_dataloader(self.flags_obj, self.dm)
    
    def set_model(self):

        self.dm.get_feature_info()
        self.model = model.DivDGLGCN(self.flags_obj, self.dm)
    
    def set_gm(self):

        self.gm = utils.DGLGraphManager(self.flags_obj, self.dm)

    def inference(self, sample):

        user, item, label_interaction, label_feature = sample
        item = item + self.dm.n_user

        data_flows, node_index, user_index, item_index = self.gm.point_sample_neighbors(user, item)

        user = user.to(self.device)
        item = item.to(self.device)
        label_interaction = label_interaction.to(self.device)
        label_feature = label_feature.to(self.device)

        score_interaction, score_feature = self.model(data_flows, node_index, user_index, item_index)

        return score_interaction, score_feature, label_interaction, label_feature

    def get_embeddings(self):

        node_flow = self.gm.get_complete_node_flows()
        user_embeddings, item_embeddings = self.model.get_embeddings(node_flow)

        return user_embeddings, item_embeddings
