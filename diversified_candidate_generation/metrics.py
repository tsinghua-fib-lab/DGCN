#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import data_utils.loader as LOADER

import config.const as const_util

import numpy as np
from scipy.stats import entropy


class Judger(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_judger'
        self.load_features(flags_obj)
        self.metrics = flags_obj.metrics
    
    def load_features(self, flags_obj):

        loader = LOADER.CsvLoader(flags_obj)
        self.item_cate = loader.load(const_util.item_cate_feature, index_col=0)
        self.cate = self.item_cate['cid'].to_numpy()
    
    def judge(self, items, test_pos, num_test_pos):

        results = {metric: 0.0 for metric in self.metrics}
        stat = self.stat(items)
        for metric in self.metrics:
            f = Metrics.get_metrics(metric)
            results[metric] = sum([f(items[i], test_pos=test_pos[i], num_test_pos=num_test_pos[i].item(), count=stat[i]) for i in range(len(items))])
        
        return results
    
    def stat(self, items):

        stat = [np.unique(self.cate[item], return_counts=True)[1] for item in items]

        return stat


class Metrics(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_metrics'

    @staticmethod
    def get_metrics(metric):

        metrics_map = {
            'recall': Metrics.recall,
            'hit_ratio': Metrics.hr,
            'coverage': Metrics.coverage,
            'entropy': Metrics.entropy,
            'gini_index': Metrics.gini,
        }

        return metrics_map[metric]

    @staticmethod
    def recall(items, **kwargs):

        test_pos = kwargs['test_pos']
        num_test_pos = kwargs['num_test_pos']
        hit_count = np.isin(items, test_pos).sum()

        return hit_count/num_test_pos

    @staticmethod
    def hr(items, **kwargs):

        test_pos = kwargs['test_pos']
        hit_count = np.isin(items, test_pos).sum()

        if hit_count > 0:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def coverage(items, **kwargs):

        count = kwargs['count']

        return count.size

    @staticmethod
    def entropy(items, **kwargs):

        count = kwargs['count']

        return entropy(count)

    @staticmethod
    def gini(items, **kwargs):

        count = kwargs['count']
        count = np.sort(count)
        n = len(count)
        cum_count = np.cumsum(count)

        return (n + 1 - 2 * np.sum(cum_count) / cum_count[-1]) / n
