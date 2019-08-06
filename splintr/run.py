'''
Author : Clarence Mah

This module defines classes to evaluate hyperparameters cleanly.
'''
import os
import time
import json
import io

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch import nn
from skorch.callbacks import Callback
from pybedtools import BedTool
from Bio.Seq import Seq
from collections import OrderedDict, namedtuple
from itertools import product
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output

import scikitplot as skplt

def show_sample(sample, class_names):
    '''
    sample (sequences, label)
        sequences (torch.Tensor) : sequences should be a tensor of [4 x 4 x N] (sequence_feature x ACTG_channels x sequence_length)
        label (torch.Tensor, int) : numerical representation of class label
    '''
    sequences, label = sample
    fig, axes = plt.subplots(4,1, sharex=True, figsize=(15,3.5))
    plt.suptitle(f'Class: {class_names[1][label]}', size=16)
    channel_names = ['Upstream Exon 3p', 'Exon 5p', 'Exon 3p', 'Downstream Exon 5p']
    feature_names = 'ACGT'
    for seq, title, ax in zip(sequences, feature_names, axes):
        ax.set_title(title, horizontalalignment='right')
        ax.title.set_position([-0.15, 0.4])
        ax.imshow(seq, cmap='binary', aspect=4, origin="")
        ax.set_yticks(range(4))
        ax.set_yticklabels(channel_names)

class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_train_loss = 0
        self.epoch_valid_loss = 0
        self.epoch_train_num_correct = 0
        self.epoch_valid_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_train_preds = []
        self.run_train_labels = []
        self.run_valid_preds = []
        self.run_valid_labels = []        
        self.run_start_time = None
        
        self.network = None
        self.train_loader = None
        self.valid_loader = None
        self.tb = None
        
    def begin_run(self, run, network, train_loader, valid_loader, log_dir=None):
        
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        
        self.network = network.cpu()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.tb = SummaryWriter(log_dir=f'{log_dir}/{run}')
        
        sequences, labels = next(iter(self.train_loader))
        
        self.tb.add_graph(self.network, sequences)
        
    def end_run(self, train_class_names, valid_class_names):
        train_fig = self._get_confusion_matrix(train_class_names, train=True)
        self.tb.add_figure(tag='train_confusion', figure=train_fig)
        
        valid_fig = self._get_confusion_matrix(valid_class_names, train=False)
        self.tb.add_figure(tag='valid_confusion', figure=valid_fig)
        
        self.tb.close()
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_train_loss = 0
        self.epoch_valid_loss = 0
        self.epoch_train_num_correct = 0
        self.epoch_valid_num_correct = 0
        
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        train_loss = self.epoch_train_loss / len(self.train_loader.dataset)
        valid_loss = self.epoch_valid_loss / len(self.valid_loader.dataset)
        
        train_accuracy = self.epoch_train_num_correct / len(self.train_loader.dataset)
        valid_accuracy = self.epoch_valid_num_correct / len(self.valid_loader.dataset)

        self.tb.add_scalar('Train Loss', train_loss, self.epoch_count)
        self.tb.add_scalar('Valid Loss', valid_loss, self.epoch_count)
        self.tb.add_scalar('Train Accuracy', train_accuracy, self.epoch_count)
        self.tb.add_scalar('Valid Accuracy', valid_accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['train_loss'] = train_loss
        results['valid_loss'] = valid_loss
        results["train_accuracy"] = train_accuracy
        results["valid_accuracy"] = valid_accuracy        
        results['epoch_duration'] = epoch_duration
        results['run_duration'] = run_duration
        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        clear_output(wait=True)
        display(df)
    
    def track_train_loss(self, loss):
        self.epoch_train_loss += loss.item() * self.train_loader.batch_size

    def track_valid_loss(self, loss):
        self.epoch_valid_loss += loss.item() * self.valid_loader.batch_size
    
    def track_train_num_correct(self, preds, labels):
        self.epoch_train_num_correct += self._get_num_correct(preds, labels)
        self.run_train_preds.extend(preds.argmax(dim=1).cpu())
        self.run_train_labels.extend(labels.cpu())

    def track_valid_num_correct(self, preds, labels):
        self.epoch_valid_num_correct += self._get_num_correct(preds, labels)
        self.run_valid_preds.extend(preds.argmax(dim=1).cpu())
        self.run_valid_labels.extend(labels.cpu())
        
    @torch.no_grad()
    def _get_confusion_matrix(self, class_names, train=True):
        class_dict = dict(enumerate(class_names))
        
        if train:
            labels = self.run_train_labels
            preds = self.run_train_preds
        else:
            labels = self.run_valid_labels
            preds = self.run_valid_preds
            
        ax = skplt.metrics.plot_confusion_matrix(y_true=[class_dict[int(l)] for l in labels],
                                                 y_pred=[class_dict[int(y)] for y in preds],
                                                 normalize=True,
                                                 hide_counts=True,
                                                 x_tick_rotation=45,
                                                 figsize=(15,15))
        return ax.get_figure()
        
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, fileName):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
    
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs