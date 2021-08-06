#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import torch

import faiss


class CandidateGenerator(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cg'
    
    def generate(self, user, k):

        raise NotImplementedError


class FaissInnerProductMaximumSearchGenerator(CandidateGenerator):

    def __init__(self, flags_obj, items):

        super(FaissInnerProductMaximumSearchGenerator, self).__init__(flags_obj)
        self.items = items
        self.embedding_size = items.shape[1]
        self.make_index(flags_obj)

    def make_index(self, flags_obj):

        self.make_index_brute_force(flags_obj)

        if flags_obj.use_gpu:

            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, flags_obj.cg_gpu_id, self.index)

    def make_index_brute_force(self, flags_obj):

        self.index = faiss.IndexFlatIP(self.embedding_size)
        self.index.add(self.items)

    def generate(self, users, k):

        _, I = self.index.search(users, k)

        return I

    def generate_with_distance(self, users, k):

        D, I = self.index.search(users, k)

        return D, I
