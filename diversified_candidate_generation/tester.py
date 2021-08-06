#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import data
import metrics

import numpy as np
import torch

from tqdm import tqdm


class Tester(object):

    def __init__(self, flags_obj, recommender):

        self.name = flags_obj.name + '_tester'
        self.recommender = recommender
        self.flags_obj = flags_obj
        self.set_topk(flags_obj)
        self.set_judger(flags_obj)
    
    def set_metrics(self, metrics):

        self.judger.metrics = metrics
    
    def set_dataloader(self, test_data_source):

        self.dataloader, self.topk_margin = data.CGDataProcessor.get_dataloader(self.flags_obj, test_data_source)
        self.n_user = self.recommender.dm.n_user
        self.cg_topk = self.max_topk + self.topk_margin
    
    def set_topk(self, flags_obj):

        self.topk = flags_obj.topk
        self.max_topk = max(flags_obj.topk)
    
    def set_judger(self, flags_obj):

        self.judger = metrics.Judger(flags_obj)
    
    def test(self):

        with torch.no_grad():

            self.init_results()
            self.make_cg()


            for data in tqdm(self.dataloader):

                users, train_pos, test_pos, num_test_pos = data
                users = users.squeeze()

                items = self.recommender.cg(users, self.cg_topk)

                items = self.filter_history(items, train_pos)

                batch_results = self.judger.judge(items, test_pos, num_test_pos)

                self.update_results(batch_results)

        self.average_user()

        return self.results
    
    def init_results(self):

        self.results = {k: 0.0 for k in self.judger.metrics}
    
    def make_cg(self):

        self.recommender.make_cg()
    
    def filter_history(self, items, train_pos):

        return np.stack([items[i][np.isin(items[i], train_pos[i], invert=True)][:self.max_topk] for i in range(len(items))], axis=0)
    
    def update_results(self, batch_results):

        for metric, value in batch_results.items():
            self.results[metric] = self.results[metric] + value
    
    def average_user(self):

        num_test_users = self.n_user
        
        for metric, value in self.results.items():
            self.results[metric] = value/num_test_users
