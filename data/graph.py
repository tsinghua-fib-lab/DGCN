#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import sys
sys.path.append('/home/zhengyu/workspace/dps')
import data_process_service.utils as utils

from tqdm import tqdm

import numpy as np
import scipy.sparse as sp



def generate_adj_graph(flags_obj):

    flags_obj.load_path = flags_obj.save_path
    loader = utils.LOADER.CooLoader(flags_obj)
    train_coo_record = loader.load('train_coo_record.npz')

    num_record = train_coo_record.nnz
    num_user = train_coo_record.shape[0]
    num_item = train_coo_record.shape[1]

    values = np.ones(2*num_record)
    row = train_coo_record.row
    col = train_coo_record.col

    col = col + num_user

    bi_row = np.hstack([row, col])
    bi_col = np.hstack([col, row])

    train_coo_adj_graph = sp.coo_matrix((values, (bi_row, bi_col)), shape=(num_user+num_item, num_user+num_item))

    saver = utils.SAVER.CooSaver(flags_obj)
    saver.save('train_coo_adj_graph.npz', train_coo_adj_graph)


def generate_edge_sim(flags_obj):

    flags_obj.load_path = flags_obj.save_path
    loader = utils.LOADER.CooLoader(flags_obj)
    train_coo_record = loader.load('train_coo_record.npz')

    num_record = train_coo_record.nnz
    num_user = train_coo_record.shape[0]
    num_item = train_coo_record.shape[1]

    loader = utils.LOADER.CsvLoader(flags_obj)
    item_cate = loader.load('item_cate_feature.csv', index_col=0)
    cate = item_cate['cid'].to_numpy()

    train_lil_record = train_coo_record.tolil()
    for u in tqdm(range(num_user)):
        items = train_lil_record.rows[u]
        cates = cate[items]
        unique, counts = np.unique(cates, return_counts=True)
        count_map = dict(zip(unique, counts))
        sim = [1/count_map[c] for c in cates]
        train_lil_record.data[u] = sim

    sim_coo_record = train_lil_record.tocoo()
    row = sim_coo_record.row
    col = sim_coo_record.col
    data = sim_coo_record.data

    col = col + num_user

    bi_row = np.hstack([row, col])
    bi_col = np.hstack([col, row])
    values = np.hstack([data, np.ones(num_record)])

    train_coo_edge_sim = sp.coo_matrix((values, (bi_row, bi_col)), shape=(num_user+num_item, num_user+num_item))

    saver = utils.SAVER.CooSaver(flags_obj)
    saver.save('train_coo_edge_sim.npz', train_coo_edge_sim)

