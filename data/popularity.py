#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import sys
sys.path.append('/home/zhengyu/workspace/dps')
import data_process_service.utils as utils

import numpy as np
import pandas as pd


def compute_popularity(flags_obj):

    flags_obj.load_path = flags_obj.save_path
    loader = utils.LOADER.CooLoader(flags_obj)
    train_coo_record = loader.load('train_coo_record.npz')

    popularity = np.zeros(train_coo_record.shape[1], dtype=np.int64)
    train_dok_record = train_coo_record.todok()
    df = pd.DataFrame(list(train_dok_record.keys()), columns=['uid', 'iid'])
    df = df.groupby('iid').count().reset_index().rename(columns={'uid': 'count'})
    popularity[df['iid']] = df['count']

    saver = utils.SAVER.NpySaver(flags_obj)
    saver.save('popularity.npy', popularity)

