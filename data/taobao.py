#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import sys
sys.path.append('/home/zhengyu/workspace/dps')
import data_process_service.utils as utils

from absl import app
from absl import flags

import popularity
import graph

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'taobao_preprocess', 'Test name.')
flags.DEFINE_bool('test', False, 'Whether in test mode.')
flags.DEFINE_string('load_path', '/home/zhengyu/data/taobao', 'Path to load file.')
flags.DEFINE_string('save_path', './taobao/', 'Path to save file.')


def filter_items_with_multiple_cids_taobao_ctr(flags_obj, record):

    item_cate = record[['iid', 'cid']].drop_duplicates().groupby('iid').count().reset_index().rename(columns={'cid': 'count'})
    items_with_single_cid = item_cate[item_cate['count'] == 1]['iid']

    id_filter = utils.FILTER.IDFilter(flags_obj, record)
    record = id_filter.filter(record, 'iid', items_with_single_cid)

    return record


def process_taobao(flags_obj):

    record = utils.load_csv(flags_obj, 'UserBehavior.csv', header=None, names=['uid', 'iid', 'cid', 'behavior', 'ts'])
    record = utils.filter_duplication(flags_obj, record)
    record = filter_items_with_multiple_cids_taobao_ctr(flags_obj, record)
    record = utils.downsample_user(flags_obj, record, 0.05)
    record = utils.filter_cf(flags_obj, record, 10)
    record, user_reindex_map, item_reindex_map = utils.reindex_user_item(flags_obj, record)
    record, cate_reindex_map = utils.reindex_feature(flags_obj, record, 'cid')
    utils.save_reindex_user_item_map(flags_obj, user_reindex_map, item_reindex_map)
    utils.save_reindex_feature_map(flags_obj, 'cate', cate_reindex_map)
    train_record, val_record, test_record = utils.split(flags_obj, record, [0.6, 0.2, 0.2])
    utils.save_csv_record(flags_obj, record, train_record, val_record, test_record)
    utils.report(flags_obj, record)
    utils.extract_save_item_feature(flags_obj, record, 'cate', 'cid')
    coo_record, train_coo_record, val_coo_record, test_coo_record = utils.generate_coo(flags_obj, record, train_record, val_record, test_record)
    utils.save_coo(flags_obj, coo_record, train_coo_record, val_coo_record, test_coo_record)


def main(argv):

    flags_obj = flags.FLAGS

    process_taobao(flags_obj)
    popularity.compute_popularity(flags_obj)
    graph.generate_adj_graph(flags_obj)
    graph.generate_edge_sim(flags_obj)


if __name__ == "__main__":

    app.run(main)

