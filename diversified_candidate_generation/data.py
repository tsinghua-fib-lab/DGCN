#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import flags

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader

import data_utils.loader as LOADER
import data_utils.sampler as SAMPLER
import data_utils.transformer as TRANSFORMER

import config.const as const_util
import utils


class FactorizationDataProcessor(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_fdp'
    
    @staticmethod
    def get_point_dataloader(flags_obj, dm):

        dataset = PointFactorizationDataset(flags_obj, dm)

        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle, num_workers=flags_obj.num_workers, drop_last=True)

    @staticmethod
    def get_adv_dataloader(flags_obj, dm):

        dataset = AdvFactorizationDataset(flags_obj, dm)

        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=flags_obj.shuffle, num_workers=flags_obj.num_workers, drop_last=True)


class FactorizationDataset(Dataset):

    def __init__(self, flags_obj, dm):

        self.name = flags_obj.name + '_dataset'
        self.make_sampler(flags_obj, dm)
    
    def make_sampler(self, flags_obj, dm):

        train_coo_record = dm.coo_record

        transformer = TRANSFORMER.SparseTransformer(flags_obj)
        train_lil_record = transformer.coo2lil(train_coo_record)
        train_dok_record = transformer.coo2dok(train_coo_record)

        self.make_sampler_core(flags_obj, train_lil_record, train_dok_record)

    def __len__(self):

        return len(self.sampler.record)

    def __getitem__(self, index):

        raise NotImplementedError


class PointFactorizationDataset(FactorizationDataset):

    def __init__(self, flags_obj, dm):

        super(PointFactorizationDataset, self).__init__(flags_obj, dm)
    
    def make_sampler_core(self, flags_obj, train_lil_record, train_dok_record):

        if not flags_obj.sim_sample:
            self.sampler = SAMPLER.PointSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)
        else:
            self.sampler = utils.SimPointSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)

    def __getitem__(self, index):

        users, items, labels = self.sampler.sample(index)

        return users, items, labels


class AdvFactorizationDataset(FactorizationDataset):

    def __init__(self, flags_obj, dm):

        super(AdvFactorizationDataset, self).__init__(flags_obj, dm)
        self.item_cate = dm.item_cate_feature
        self.cate = self.item_cate['cid'].to_numpy()

    def make_sampler_core(self, flags_obj, train_lil_record, train_dok_record):

        if not flags_obj.sim_sample:
            self.sampler = SAMPLER.PointSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)
        else:
            self.sampler = utils.SimPointSampler(flags_obj, train_lil_record, train_dok_record, flags_obj.neg_sample_rate)
    
    def __getitem__(self, index):

        users, items, labels_interaction = self.sampler.sample(index)
        labels_feature = self.cate[items]

        return users, items, labels_interaction, labels_feature


class CGDataProcessor(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cdp'
    
    @staticmethod
    def get_dataloader(flags_obj, test_data_source):

        dataset = CGDataset(flags_obj, test_data_source)

        return DataLoader(dataset, batch_size=flags_obj.batch_size, shuffle=True, num_workers=flags_obj.num_workers, drop_last=False), dataset.max_train_interaction


class CGDataset(Dataset):

    def __init__(self, flags_obj, test_data_source):

        self.name = flags_obj.name + '_dataset'
        self.test_data_source = test_data_source
        self.sort_users(flags_obj)
    
    def sort_users(self, flags_obj):

        loader = LOADER.CooLoader(flags_obj)
        if self.test_data_source == 'val':
            coo_record = loader.load(const_util.val_coo_record)
        elif self.test_data_source == 'test':
            coo_record = loader.load(const_util.test_coo_record)
        transformer = TRANSFORMER.SparseTransformer(flags_obj)
        self.lil_record = transformer.coo2lil(coo_record)

        train_coo_record = loader.load(const_util.train_coo_record)
        self.train_lil_record = transformer.coo2lil(train_coo_record)

        train_interaction_count = np.array([len(row) for row in self.train_lil_record.rows], dtype=np.int64)
        self.max_train_interaction = int(max(train_interaction_count))

        test_interaction_count = np.array([len(row) for row in self.lil_record.rows], dtype=np.int64)
        self.max_test_interaction = int(max(test_interaction_count))
    
    def __len__(self):

        return len(self.lil_record.rows)
    
    def __getitem__(self, index):

        unify_train_pos = np.full(self.max_train_interaction, -1, dtype=np.int64)
        unify_test_pos = np.full(self.max_test_interaction, -1, dtype=np.int64)

        train_pos = self.train_lil_record.rows[index]
        test_pos = self.lil_record.rows[index]

        unify_train_pos[:len(train_pos)] = train_pos
        unify_test_pos[:len(test_pos)] = test_pos

        return torch.LongTensor([index]), torch.LongTensor(unify_train_pos), torch.LongTensor(unify_test_pos), torch.LongTensor([len(test_pos)])

