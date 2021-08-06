#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import os
import datetime
import setproctitle

from absl import logging
from visdom import Visdom

from tqdm import tqdm

import numpy as np
import scipy.sparse as sp

import torch
import dgl

import config.const as const_util
import trainer
import recommender

import data_utils.loader as LOADER
import data_utils.transformer as TRANSFORMER
import data_utils.sampler as SAMPLER


class ContextManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_cm'
        self.exp_name = flags_obj.name
        self.output = flags_obj.output
    
    def set_default_ui(self):

        self.set_workspace()
        self.set_process_name()
        self.set_logging()
    
    def set_workspace(self):

        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        dir_name = self.exp_name + '_' + date_time
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        self.workspace = os.path.join(self.output, dir_name)
        os.mkdir(self.workspace)
    
    def set_process_name(self):

        setproctitle.setproctitle(self.exp_name + '@zhengyu')
    
    def set_logging(self):

        self.log_path = os.path.join(self.workspace, 'log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)
    
    def set_test_logging(self):

        self.log_path = os.path.join(self.workspace, 'test_log')
        if not os.path.exists(self.log_path):

            os.mkdir(self.log_path)

        logging.flush()
        logging.get_absl_handler().use_absl_log_file(self.exp_name + '.log', self.log_path)
    
    def logging_flags(self, flags_obj):

        logging.info('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            logging.info('{}: {}'.format(flag, value))

    @staticmethod
    def set_trainer(flags_obj, cm, vm, dm):

        if flags_obj.loss == 'point':
            return trainer.PointTrainer(flags_obj, cm, vm, dm)
        elif flags_obj.loss == 'adv':
            return trainer.AdversarialTrainer(flags_obj, cm, vm, dm)

    @staticmethod
    def set_recommender(flags_obj, workspace, dm):

        if not flags_obj.loss == 'adv':
            return recommender.GCNRecommender(flags_obj, workspace, dm)
        else:
            return recommender.DGCNRecommender(flags_obj, workspace, dm)
    
    @staticmethod
    def set_device(flags_obj):

        if not flags_obj.use_gpu:
            return torch.device('cpu')       
        else:
            return torch.device('cuda:{}'.format(flags_obj.gpu_id))


class VizManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_vm'
        self.exp_name = flags_obj.name
        self.port = flags_obj.port
        self.set_visdom()
    
    def set_visdom(self):

        self.viz = Visdom(port=self.port, env=self.exp_name)
    
    def get_new_text_window(self, title):

        win = self.viz.text(title)

        return win
    
    def append_text(self, text, win):

        self.viz.text(text, win=win, append=True)
    
    def show_basic_info(self, flags_obj):

        basic = self.viz.text('Basic Information:')
        self.viz.text('Name: {}'.format(flags_obj.name), win=basic, append=True)
        self.viz.text('Model: {}'.format(flags_obj.model), win=basic, append=True)
        self.viz.text('Dataset: {}'.format(flags_obj.dataset), win=basic, append=True)
        self.viz.text('Embedding Size: {}'.format(flags_obj.embedding_size), win=basic, append=True)
        self.viz.text('Initial lr: {}'.format(flags_obj.lr), win=basic, append=True)
        self.viz.text('Batch Size: {}'.format(flags_obj.batch_size), win=basic, append=True)
        self.viz.text('Weight Decay: {}'.format(flags_obj.weight_decay), win=basic, append=True)
        self.viz.text('Negative Sampling Ratio: {}'.format(flags_obj.neg_sample_rate), win=basic, append=True)

        self.basic = basic

        flags = self.viz.text('FLAGS:')
        for flag, value in flags_obj.flag_values_dict().items():
            self.viz.text('{}: {}'.format(flag, value), win=flags, append=True)

        self.flags = flags
    
    def show_test_info(self, flags_obj):

        test = self.viz.text('Test Information:')
        self.test = test
    
    def step_update_line(self, title, value):

        if not hasattr(self, title):

            setattr(self, title, self.viz.line([value], [0], opts=dict(title=title)))
            setattr(self, title + '_step', 1)
        
        else:

            step = getattr(self, title + '_step')
            self.viz.line([value], [step], win=getattr(self, title), update='append')
            setattr(self, title + '_step', step + 1)
    
    def step_update_multi_lines(self, kv_record):

        for title, value in kv_record.items():
            self.step_update_line(title, value)
    
    def show_result(self, results):

        self.viz.text('-----Results-----', win=self.test, append=True)

        for metric, value in results.items():
            
            self.viz.text('{}: {}'.format(metric, value), win=self.test, append=True)
        
        self.viz.text('-----------------', win=self.test, append=True)


class DatasetManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_dm'
        self.make_coo_loader_transformer(flags_obj)
        self.make_npy_loader(flags_obj)
        self.make_csv_loader(flags_obj)

    def make_coo_loader_transformer(self, flags_obj):

        self.coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_transformer = TRANSFORMER.SparseTransformer(flags_obj)

    def make_npy_loader(self, flags_obj):

        self.npy_loader = LOADER.NpyLoader(flags_obj)

    def make_csv_loader(self, flags_obj):

        self.csv_loader = LOADER.CsvLoader(flags_obj)

    def get_dataset_info(self):

        coo_record = self.coo_loader.load(const_util.train_coo_record)

        self.n_user = coo_record.shape[0]
        self.n_item = coo_record.shape[1]

        self.coo_record = coo_record

    def get_feature_info(self):

        if not hasattr(self, 'item_cate_feature'):
            self.get_feature_info_core()

    def get_feature_info_core(self):

        item_cate_feature = self.csv_loader.load(const_util.item_cate_feature, index_col=0)
        self.n_feature = item_cate_feature['cid'].nunique()

        self.item_cate_feature = item_cate_feature

    def get_popularity(self):

        popularity = self.npy_loader.load(const_util.popularity)
        return popularity


class EarlyStopManager(object):

    def __init__(self, flags_obj):

        self.name = flags_obj.name + '_esm'
        self.min_lr = flags_obj.min_lr
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, lr, metric):

        if lr > self.min_lr:
            if metric > self.max_metric:
                self.max_metric = metric
            return False
        else:
            if metric > self.max_metric:
                self.max_metric = metric
                self.count = 0
                return False
            else:
                self.count = self.count + 1
                if self.count > self.es_patience:
                    return True
                return False


class DGLGraphManager(object):

    def __init__(self, flags_obj, dm):

        self.name = flags_obj.name + '_gm'
        self.make_coo_loader_transformer(flags_obj)
        self.sim_aggregate = flags_obj.sim_aggregate
        self.edge_sim_alpha = flags_obj.edge_sim_alpha
        self.load_graph(flags_obj, dm)
        self.hop = flags_obj.hop
        self.num_sample = flags_obj.sample_neighbor
        self.device = ContextManager.set_device(flags_obj)

    def make_coo_loader_transformer(self, flags_obj):

        self.coo_loader = LOADER.CooLoader(flags_obj)
        self.coo_transformer = TRANSFORMER.SparseTransformer(flags_obj)

    def load_graph(self, flags_obj, dm):

        self.coo_adj_graph = self.coo_loader.load(const_util.train_coo_adj_graph)

        self.graph = dgl.DGLGraph()

        num_nodes = self.coo_adj_graph.shape[0]
        self.graph.add_nodes(num_nodes)
        self.graph.ndata['feature'] = torch.arange(num_nodes)

        self.graph.add_edges(self.coo_adj_graph.row, self.coo_adj_graph.col)
        self.graph.add_edges(self.graph.nodes(), self.graph.nodes())

        if flags_obj.sim_aggregate:
            self.add_edge_sim(dm)

        self.graph.readonly()

    def add_edge_sim(self, dm):

        dm.get_feature_info()

        num_user = dm.n_user
        num_item = dm.n_item
        num_record = dm.coo_record.nnz
        cate = dm.item_cate_feature['cid'].to_numpy()

        train_lil_record = dm.coo_record.tolil()
        for u in tqdm(range(num_user)):
            items = train_lil_record.rows[u]
            cates = cate[items]
            unique, counts = np.unique(cates, return_counts=True)
            count_map = dict(zip(unique, counts))
            sim = [1/(count_map[c]**self.edge_sim_alpha) for c in cates]
            train_lil_record.data[u] = sim

        sim_coo_record = train_lil_record.tocoo()
        row = sim_coo_record.row
        col = sim_coo_record.col
        data = sim_coo_record.data

        col = col + num_user

        bi_row = np.hstack([row, col])
        bi_col = np.hstack([col, row])
        values = np.hstack([data, np.ones(num_record)])

        edge_sim = sp.coo_matrix((values, (bi_row, bi_col)), shape=(num_user+num_item, num_user+num_item))
        self.graph.edges[edge_sim.row, edge_sim.col].data['sim'] = torch.FloatTensor(edge_sim.data)

        self.graph.edges[self.graph.nodes(), self.graph.nodes()].data['sim'] = torch.ones(len(self.graph.nodes()))

    def point_sample_neighbors(self, user, item):

        seed_nodes, node_indices = self.generate_seed_nodes([user, item])
        node_flow = self.sample_neighbors(seed_nodes)

        return node_flow, None, node_indices[0], node_indices[1]

    def generate_seed_nodes(self, nodes):

        nodes_np = [node.numpy() for node in nodes]

        shape = nodes_np[0].shape

        nodes_list = [node.reshape(-1) for node in nodes_np]

        seed_nodes = list(set.union(*[set(node) for node in nodes_list]))

        nodes_map = {n:i for i, n in enumerate(seed_nodes)}
        node_indices = [torch.LongTensor(np.array([nodes_map[u] for u in node]).reshape(shape)) for node in nodes_list]

        return seed_nodes, node_indices

    def sample_neighbors(self, seed_nodes):

        if not self.sim_aggregate:
            node_flow = dgl.contrib.sampling.NeighborSampler(g=self.graph,
                                                             batch_size=len(seed_nodes),
                                                             expand_factor=self.num_sample,
                                                             neighbor_type='in',
                                                             num_hops=self.hop,
                                                             seed_nodes=seed_nodes,
                                                             add_self_loop=True)
        else:
            node_flow = dgl.contrib.sampling.NeighborSampler(g=self.graph,
                                                             batch_size=len(seed_nodes),
                                                             expand_factor=self.num_sample,
                                                             neighbor_type='in',
                                                             num_hops=self.hop,
                                                             transition_prob='sim',
                                                             seed_nodes=seed_nodes,
                                                             add_self_loop=True)

        return node_flow

    def get_complete_node_flows(self):

        if not self.sim_aggregate:
            node_flow = dgl.contrib.sampling.NeighborSampler(g=self.graph,
                                                             batch_size=len(self.graph.nodes()),
                                                             expand_factor=self.num_sample,
                                                             neighbor_type='in',
                                                             num_hops=self.hop,
                                                             seed_nodes=self.graph.nodes(),
                                                             add_self_loop=True)
        else:
            node_flow = dgl.contrib.sampling.NeighborSampler(g=self.graph,
                                                             batch_size=len(self.graph.nodes()),
                                                             expand_factor=self.num_sample,
                                                             neighbor_type='in',
                                                             num_hops=self.hop,
                                                             transition_prob='sim',
                                                             seed_nodes=self.graph.nodes(),
                                                             add_self_loop=True)

        return node_flow


class SimPointSampler(SAMPLER.Sampler):

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate):

        super(SimPointSampler, self).__init__(flags_obj, lil_record, dok_record, neg_sample_rate)

        self.sim_prob_beta = flags_obj.sim_prob_beta
        self.load_sim(flags_obj)

    def load_sim(self, flags_obj):

        loader = LOADER.CsvLoader(flags_obj)
        item_cate = loader.load(const_util.item_cate_feature, index_col=0)
        self.cate = item_cate['cid'].to_numpy()
        self.cate_items = item_cate.groupby('cid')['iid'].apply(np.array).to_numpy()

    def sample(self, index):

        user, pos_item = self.get_pos_user_item(index)

        users = np.full(1 + self.neg_sample_rate, user, dtype=np.int64)
        items = np.full(1 + self.neg_sample_rate, pos_item, dtype=np.int64)
        labels = np.zeros(1 + self.neg_sample_rate, dtype=np.float32)

        labels[0] = 1.0

        negative_samples = self.generate_negative_samples(user, pos_item=pos_item)
        items[1:] = negative_samples[:]

        return users, items, labels

    def generate_negative_samples(self, user, **kwargs):

        pos_item = kwargs['pos_item']
        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)

        user_pos = self.lil_record.rows[user]

        sim_items = self.get_sim_items(pos_item)
        sim_items = sim_items[np.logical_not(np.isin(sim_items, user_pos))]
        num_sim_items = len(sim_items)

        for count in range(self.neg_sample_rate):

            if num_sim_items > 0 and np.random.random() < self.sim_prob_beta:
                index = np.random.randint(num_sim_items)
                item = sim_items[index]
            else:
                item = np.random.randint(self.n_item)
                while item in user_pos or item in negative_samples:
                    item = np.random.randint(self.n_item)

            negative_samples[count] = item

        return negative_samples

    def get_sim_items(self, item):

        sim_items = self.cate_items[self.cate[item]]
        return sim_items
