#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import app
from absl import flags

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('name', 'DGCN-debug', 'Experiment name.')
flags.DEFINE_string('model', 'DGCN', 'Model name.')
flags.DEFINE_bool('use_gpu', True, 'Use GPU or not.')
flags.DEFINE_integer('gpu_id', 6, 'GPU ID.')
flags.DEFINE_integer('cg_gpu_id', 6, 'GPU ID for candidate generation.')
flags.DEFINE_string('dataset', 'taobao', 'Dataset.')
flags.DEFINE_integer('embedding_size', 32, 'Embedding size for embedding based models.')
flags.DEFINE_integer('epochs', 200, 'Max epochs for training.')
flags.DEFINE_bool('sim_sample', False, 'Use SimSampler or not.')
flags.DEFINE_float('sim_prob_beta', 0.5, 'Probability to choose similar items as negative samples')
flags.DEFINE_bool('sim_aggregate', False, 'Whether to perform neighbor aggregation according to similarity or not.')
flags.DEFINE_float('edge_sim_alpha', 1.0, 'Exponent of edge sim')
flags.DEFINE_float('lr', 0.01, 'Learning rate.')
flags.DEFINE_float('min_lr', 0.0001, 'Minimum learning rate.')
flags.DEFINE_float('weight_decay', 5e-8, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout ratio.')
flags.DEFINE_integer('batch_size', 2048, 'Batch Size.')
flags.DEFINE_enum('loss', 'point', ['point', 'adv'], 'Loss function.')
flags.DEFINE_float('adv_gamma', 0.1, 'Weight for adversarial loss.')
flags.DEFINE_integer('neg_sample_rate', 4, 'Negative Sampling Ratio.')
flags.DEFINE_integer('hop', 1, 'Number of hops to perform graph convolution.')
flags.DEFINE_integer('sample_neighbor', 20, 'Number of neighbors to sample.')
flags.DEFINE_bool('shuffle', True, 'Shuffle the training set or not.')
flags.DEFINE_multi_string('metrics', ['recall', 'hit_ratio', 'coverage', 'entropy', 'gini_index'], 'Metrics.')
flags.DEFINE_multi_string('val_metrics', ['recall', 'hit_ratio', 'coverage', 'entropy', 'gini_index'], 'Metrics.')
flags.DEFINE_string('watch_metric', 'recall', 'Which metric to decide learning rate reduction.')
flags.DEFINE_integer('patience', 5, 'Patience for reducing learning rate.')
flags.DEFINE_integer('es_patience', 3, 'Patience for early stop.')
flags.DEFINE_enum('test_model', 'best', ['best', 'last'], 'Which model to test.')
flags.DEFINE_multi_integer('topk', [300], 'Topk for testing recommendation performance.')
flags.DEFINE_integer('num_workers', 8, 'Number of processes for training and testing.')
flags.DEFINE_string('load_path', '', 'Load path.')
flags.DEFINE_string('output', '', 'Directory to save model/log/metrics.')
flags.DEFINE_integer('port', 33333, 'Port to show visualization results.')


def main(argv):

    flags_obj = FLAGS
    cm = utils.ContextManager(flags_obj)
    vm = utils.VizManager(flags_obj)
    dm = utils.DatasetManager(flags_obj)
    dm.get_dataset_info()

    cm.set_default_ui()
    cm.logging_flags(flags_obj)
    vm.show_basic_info(flags_obj)
    trainer = utils.ContextManager.set_trainer(flags_obj, cm, vm, dm)
    trainer.train()

    trainer.test()


if __name__ == "__main__":
    
    app.run(main)
