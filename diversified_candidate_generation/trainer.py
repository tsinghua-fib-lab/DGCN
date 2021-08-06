#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

from absl import logging

import time
from tqdm import tqdm

import utils
from tester import Tester

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Trainer(object):

    def __init__(self, flags_obj, cm, vm, dm):

        self.name = flags_obj.name + '_trainer'
        self.cm = cm
        self.vm = vm
        self.dm = dm
        self.flags_obj = flags_obj
        self.lr = flags_obj.lr
        self.set_recommender(flags_obj, cm.workspace, dm)
        self.recommender.transfer_model()
        self.tester = Tester(flags_obj, self.recommender)
    
    def set_recommender(self, flags_obj, workspace, dm):

        self.recommender = utils.ContextManager.set_recommender(flags_obj, workspace, dm)
    
    def train(self):

        self.set_dataloader()
        self.tester.set_dataloader('val')
        self.tester.set_metrics(self.flags_obj.val_metrics)
        self.set_optimizer()
        self.set_scheduler()
        self.set_esm()
        self.set_leaderboard()

        for epoch in range(self.flags_obj.epochs):

            self.train_one_epoch(epoch)
            watch_metric_value = self.validate(epoch)
            self.recommender.save_ckpt(epoch)
            self.scheduler.step(watch_metric_value)
            self.update_leaderboard(epoch, watch_metric_value)
            
            stop = self.esm.step(self.lr, watch_metric_value)
            if stop:
                break
                    
    def test(self):

        self.cm.set_test_logging()
        self.tester.set_dataloader('test')
        self.tester.set_metrics(self.flags_obj.metrics)

        if self.flags_obj.test_model == 'best':
            self.recommender.load_ckpt(self.max_epoch)
            logging.info('best epoch: {}'.format(self.max_epoch))

        self.vm.show_test_info(self.flags_obj)

        for topk in self.flags_obj.topk:

            self.tester.max_topk = topk
            results = self.tester.test()
            self.vm.show_result(results)
            
            logging.info('TEST results: topk = {}'.format(topk))
            for metric, value in results.items():
                logging.info('{}: {}'.format(metric, value))
    
    def set_dataloader(self):

        raise NotImplementedError
    
    def set_optimizer(self):

        self.optimizer = self.recommender.get_optimizer()
    
    def set_scheduler(self):

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.flags_obj.patience, min_lr=self.flags_obj.min_lr)
    
    def set_esm(self):

        self.esm = utils.EarlyStopManager(self.flags_obj)
    
    def set_leaderboard(self):

        self.max_metric = -1.0
        self.max_epoch = -1
        self.leaderboard = self.vm.get_new_text_window('leaderboard')
    
    def update_leaderboard(self, epoch, metric):

        if metric > self.max_metric:

            self.max_metric = metric
            self.max_epoch = epoch

            self.vm.append_text('New Record! {} @ epoch {}!'.format(metric, epoch), self.leaderboard)
    
    def train_one_epoch(self, epoch):

        start_time = time.time()
        running_loss = 0.0
        total_loss = 0.0
        num_batch = len(self.dataloader)
        self.distances = np.zeros(num_batch)

        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < self.lr:

            self.lr = current_lr
            logging.info('reducing learning rate!')

        logging.info('learning rate : {}'.format(self.lr))

        for batch_count, sample in enumerate(tqdm(self.dataloader)):

            self.optimizer.zero_grad()

            loss = self.get_loss(sample, batch_count)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            if batch_count % 1000 == 0:
                self.vm.step_update_line('loss every 1k step', loss.item())

            if batch_count % (num_batch // 5) == num_batch // 5 - 1:

                logging.info('epoch {}: running loss = {}'.format(epoch, running_loss / (num_batch // 5)))
                running_loss = 0.0

        logging.info('epoch {}: total loss = {}'.format(epoch, total_loss))
        self.vm.step_update_line('epoch loss', total_loss)
        self.vm.step_update_line('distance', self.distances.mean())

        time_cost = time.time() - start_time
        self.vm.step_update_line('train time cost', time_cost)

    def get_loss(self, sample, batch_count):

        raise NotImplementedError

    def validate(self, epoch):

        start_time = time.time()
        results = self.tester.test()
        self.vm.step_update_multi_lines(results)
        logging.info('VALIDATION epoch: {}, results: {}'.format(epoch, results))
        time_cost = time.time() - start_time
        self.vm.step_update_line('validate time cost', time_cost)

        return results[self.flags_obj.watch_metric]


class PointTrainer(Trainer):

    def __init__(self, flags_obj, cm, vm, dm):

        super(PointTrainer, self).__init__(flags_obj, cm, vm, dm)
        self.criterion = nn.BCELoss()

    def set_dataloader(self):

        self.dataloader = self.recommender.get_dataloader()

    def get_loss(self, sample, batch_count):

        score, label = self.recommender.point_inference(sample)
        score = torch.sigmoid(score)

        self.distances[batch_count] = (score - label).abs().mean().item()

        loss = self.log_loss(score, label)

        return loss

    def log_loss(self, score, label):

        loss = self.criterion(score, label)
        return loss


class AdversarialTrainer(Trainer):

    def __init__(self, flags_obj, cm, vm, dm):

        super(AdversarialTrainer, self).__init__(flags_obj, cm, vm, dm)
        self.adv_gamma = flags_obj.adv_gamma
        self.criterion_interaction = nn.BCELoss()
        self.criterion_feature = nn.CrossEntropyLoss()

        self.feature_classification_total = 0
        self.feature_classification_correct = 0

    def set_dataloader(self):

        self.dataloader = self.recommender.get_dataloader()

    def get_loss(self, sample, batch_count):

        score_interaction, score_feature, label_interaction, label_feature = self.recommender.inference(sample)
        score_interaction = torch.sigmoid(score_interaction)

        self.distances[batch_count] = (score_interaction - label_interaction).abs().mean().item()

        loss_interaction = self.log_loss(score_interaction, label_interaction)
        loss_feature_classification, feature_classification_total, feature_classification_correct = self.ce_loss(score_feature, label_feature)

        loss = loss_interaction + self.adv_gamma * loss_feature_classification

        self.feature_classification_total = self.feature_classification_total + feature_classification_total
        self.feature_classification_correct = self.feature_classification_correct + feature_classification_correct

        if batch_count % 1000 == 0:

            feature_classification_acc = self.feature_classification_correct / self.feature_classification_total
            self.feature_classification_total = 0
            self.feature_classification_correct = 0

            kv_record = {
                'interaction loss every 1k step': loss_interaction.item(),
                'adversarial loss every 1k step': loss_feature_classification.item(),
                'feature classification acc': feature_classification_acc
            }
            self.vm.step_update_multi_lines(kv_record)

        return loss

    def log_loss(self, score, label):

        loss = self.criterion_interaction(score, label)
        return loss

    def ce_loss(self, score, label):

        score = score.view(-1, score.size(-1))
        label = label.view(-1)
        loss = self.criterion_feature(score, label)

        _, predicted = torch.max(score.data, 1)
        feature_classification_correct = (predicted == label).sum().item()
        feature_classification_total = label.size(0)

        return loss, feature_classification_total, feature_classification_correct

