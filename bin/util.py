import os
import time
import json
import io

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch import nn
from pybedtools import BedTool
from Bio.Seq import Seq
from collections import OrderedDict, namedtuple
from itertools import product
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output

import scikitplot as skplt

def show_sample(sample, class_names):
    '''
    sample (sequences, label)
        sequences (torch.Tensor) : sequences should be a tensor of [4 x 4 x N] (sequence_feature x ACTG_channels x sequence_length)
        label (torch.Tensor, int) : numerical reprensentation of class label
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
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_preds = []
        self.run_labels = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader, log_dir=None):
        
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        
        self.network = network.cpu()
        self.loader = loader
        self.tb = SummaryWriter(log_dir=f'{log_dir}/{run}')
        
        sequences, labels = next(iter(self.loader))
        
        self.tb.add_graph(self.network, sequences)
        
    def end_run(self, class_names):
        cfm_fig = self._get_confusion_matrix(class_names)
        self.tb.add_figure(tag='Confusion Matrix', figure=cfm_fig)
        self.tb.close()
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Cross Entropy Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        clear_output(wait=True)
        display(df)
    
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
        self.run_preds.extend(preds.argmax(dim=1).cpu())
        self.run_labels.extend(labels.cpu())
    
    @torch.no_grad()
    def _get_confusion_matrix(self, class_names):
        class_dict = dict(enumerate(class_names))
        ax = skplt.metrics.plot_confusion_matrix(y_true=[class_dict[int(l)] for l in self.run_labels],
                                                 y_pred=[class_dict[int(y)] for y in self.run_preds],
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

class ConvNet(nn.Module):
    '''
    Input tensor size: [batch_size, in_channel, height, width]
    '''
    def __init__(self,
                 num_classes,
                 c1_in,
                 c1_out,
                 c1_kernel_w,
                 c1_filter,
                 c1_stride_w,
                 c2_out,
                 c2_kernel_w,
                 c2_filter,
                 c2_stride_w,
                 fc_out):
        super().__init__()
        
        # Calculate appropriate padding to guarantee c1_out
        c1_padding = calc_conv_pad(c1_in, c1_out, c1_kernel_w, c1_stride_w)
        
        # TODO make sure padding is > 0, < ?
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=c1_filter,
                      kernel_size=(1, c1_kernel_w),
                      stride=(1, c1_stride_w),
                      padding=(0, c1_padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)))
        
        c2_padding = calc_conv_pad(c1_out/2, c2_out, c2_kernel_w, c2_stride_w)
        # TODO make sure padding is > 0, < ?
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=c1_filter,
                      out_channels=c2_filter,
                      kernel_size=(1, c2_kernel_w),
                      stride=(1, c2_stride_w),
                      padding=(0, c2_padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))   
        self.drop_out = nn.Dropout(p=0.2)
        
        fc_in = int(c2_filter*4*round(c2_out/2))
        self.fc1 = nn.Linear(in_features=fc_in, out_features=fc_out)
        self.output = nn.Sequential(
            nn.Linear(in_features=fc_out, out_features=num_classes),
            nn.Softmax())

        
    def forward(self, x):
#         print('Input: ', x.shape)
        out = self.layer1(x)
#         print('CNN1 output: ', out.shape)
        out = self.layer2(out)
#         print('CNN2 output: ', out.shape)
        out = out.reshape(out.size(0), -1)
#         print('FC input: ', out.shape)
#         out = self.drop_out(out)
        out = self.fc1(out)
#         print('FC output: ', out.shape)
        out = self.output(out)
#         print('Final output: ', out.shape)
        return out


class SpliceSeqDataset(Dataset):
    
    def __init__(self, feature_files, genome_fa, transform=None):
        '''
        feature_file (string): File with sequences and class labels.
        transform (callable, optional): Optional transform to be applied on a sample.
        '''
        self.features = []
        for file in feature_files:
            bed = BedTool(file)
            
            sequences = [line.strip().upper() for line in open(bed.getfasta(genome_fa).seqfn).readlines() if not line.startswith('>')]
            self.features.append(sequences)
            
        labels = pd.read_csv(file, sep='\t', header=None).iloc[:,3]
        self.labels = pd.factorize(labels)[0]
        
        self.transform = transform
        
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        sequences = [seq_group[idx] for seq_group in self.features]
        label = self.labels[idx]
                
        sample = {'sequences': sequences, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample['sequences'], sample['label']
    
    
class PadSequence(object):
    '''
    Generate reverse complement of sample sequences.
    '''

    def __init__(self, seq_length):
        self.seq_length = seq_length
        pass
    
    def __call__(self, sample):
        sequences = sample['sequences']
        
        new_seqs = []
        for seq in sequences:
            pad_total = self.seq_length - len(seq)
            if pad_total % 2 == 0:
                pad_left = int(pad_total / 2)
                pad_right = int(pad_total / 2)
            else:
                pad_left = round(pad_total / 2)
                pad_right = pad_total - pad_left
            new_seqs.append('N'*pad_left + seq + 'N'*pad_right)
        sample['sequences'] = new_seqs
        
        return sample  


class ReverseComplement(object):
    '''
    Generate reverse complement of sample sequences.
    '''

    def __init__(self):
        pass
    
    def __call__(self, sample):
        sequences = sample['sequences']
        
        new_seqs = []
        for seq in sequences:
            new_seqs.append(str(Seq(seq).reverse_complement()))
        sample['sequences'] = new_seqs
        
        return sample
        
class CropSequence(object):
    '''
    Generate reverse complement of sample sequence.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def __call__(self, sample):
        sequences = sample['sequences']
        
        new_seqs = []
        for seq in sequences:
            seq_len = len(seq)
            
            if seq_len != self.output_size:            
                start = np.random.randint(0, seq_len - self.output_size)
                end = start + self.output_size
                seq = seq[start:end]
                
            new_seqs.append(seq)
        sample['sequences'] = new_seqs
            
        return sample
    
class ToOneHotEncoding(object):
    '''
    Convert ndarrays
    '''
    def __init__(self):
        self.base2index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    def __call__(self, sample):
        sequences = sample['sequences']
        
        new_seqs = []
        for seq in sequences:
            one_hot = np.zeros((len(seq), 4), dtype=np.float32)
            for i, base in enumerate(seq):
                if base in self.base2index:
                    one_hot[i, self.base2index[base]] = 1.
            new_seqs.append(np.transpose(one_hot))
            
        new_seqs = np.array(new_seqs)
        new_seqs = np.transpose(new_seqs, (1, 0, 2))
            
        sample['sequences'] = new_seqs
        return sample
    
def calc_conv_pad(input_size, output_size, kernel_size, stride):
    '''
    Calculate appropriate padding to guarantee output size.
    '''
    return int((output_size * stride - input_size + kernel_size - stride) / 2)

