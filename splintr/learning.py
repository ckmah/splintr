'''
Author : Clarence Mah

This module defines the custom PyTorch neural network Module and Dataset
for the neural network classifier structure using PyTorch for performing splice-event RBP classification. 
'''
import splintr as sp
from splintr.splice import SpliceData
from splintr.util import vprint

import math 

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision.transforms import Compose

from pybedtools import BedTool
from Bio.Seq import Seq

from tqdm.autonotebook import tqdm
tqdm.pandas()
    
def _calc_conv_pad(input_size, output_size, kernel_size, stride):
    '''
    Calculate appropriate padding to guarantee output size.
    '''
    return math.ceil((output_size * stride - input_size + kernel_size - stride) / 2)

def fit(parameterization):
    net = train(parameterization)
    return validate(net)
    manager = sp.RunManager()
    is_first_run = True
    for run in sp.RunBuilder.get_runs(params):    
        # Initialize model and dataset
        network = sp.SplintrNet(num_classes=run.num_classes,
                          c1_in=run.c1_in,
                          c1_out=run.c1_out,
                          c1_kernel_w=run.c1_kernel_w,
                          c1_filter=run.c1_filter,
                          c1_stride_w=run.c1_stride_w,
                          c2_out=run.c2_out,
                          c2_kernel_w=run.c2_kernel_w,
                          c2_filter=run.c2_filter,
                          c2_stride_w=run.c2_stride_w,
                          dropout=run.dropout,
                          fc_out=run.fc_out).cuda(device)

        train_loader = DataLoader(train_dataset, batch_size=run.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, batch_size=run.batch_size, sampler=valid_sampler)

        optimizer = torch.optim.Adam(network.parameters(), lr=run.lr, weight_decay=run.weight_decay)
        log_dir = '/home/ubuntu/tb/8-05-19-6class/'
        # Display brief summary of first model
        if is_first_run:
            is_first_run = False
            summary(network.cuda(), input_size=(4, 4, seq_length), device='cuda')
    #         util.show_sample(train_dataset[np.random.randint(len(train_dataset))], class_names=label_names)

        # Perform training
        manager.begin_run(run, network, train_loader, valid_loader, log_dir)
        network.cuda()
        for epoch in range(30):

            manager.begin_epoch()

            # Train on batch
            for batch in train_loader:
                seqs, labels = batch
                preds = network(seqs.cuda(device)) # pass batch
                loss = F.cross_entropy(preds, labels.cuda(device)) # calculate loss
                optimizer.zero_grad() # zero gradients
                loss.backward() # calculate gradients
                optimizer.step() # update weights

                manager.track_train_loss(loss)
                manager.track_train_num_correct(preds, labels.cuda(device))

            # Check validation set
            with torch.no_grad():
                for data in valid_loader:
                    seqs, labels = data
                    preds = network(seqs.cuda(device))
                    loss = F.cross_entropy(preds, labels.cuda(device))

                    manager.track_valid_loss(loss)
                    manager.track_valid_num_correct(preds, labels.cuda(device))

            manager.end_epoch()
        manager.end_run(train_class_names=label_names[1],
                        valid_class_names=label_names[1])
    manager.save('../results')

class SplintrNet(nn.Module):
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
                 dropout,
                 fc_out):
        super().__init__()
        
        # Calculate appropriate padding to guarantee c1_out
        c1_padding = _calc_conv_pad(c1_in, c1_out, c1_kernel_w, c1_stride_w)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=c1_filter,
                      kernel_size=(1, c1_kernel_w),
                      stride=(1, c1_stride_w),
                      padding=(0, c1_padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)))
        
        c2_padding = _calc_conv_pad(c1_out/2, c2_out, c2_kernel_w, c2_stride_w)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=c1_filter,
                      out_channels=c2_filter,
                      kernel_size=(1, c2_kernel_w),
                      stride=(1, c2_stride_w),
                      padding=(0, c2_padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)))   
        self.drop_out = nn.Dropout(p=dropout)
        
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
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.output(out)
#         print('Final output: ', out.shape)
        return out


class SpliceEventDataset(Dataset):
    
    def __init__(self, feature_file, genome_fa, transform=None):
        '''
        feature_file (str) : path to RMATs file
        
        genome_fa (str) : human genome fasta
        
        transform (list, optional) : Optional list of transforms to be applied on a sample.
        '''
        vprint('Creating SpliceEventDataset object...')

        # Create splice object for cleaner data representation
        jc = SpliceData(feature_file)
        
        vprint('| Retrieving junction intervals...')
        event_regions = jc.get_junction_regions(50, 350) # n x 4 for SE
        
        vprint('| Retrieving fasta sequences...')
        # First convert to string for BedTools
        bed_str = ''
        for regions in event_regions:
            for region in regions:
                bed_str += ' '.join([str(s) for s in region])
                bed_str += '\n'            

        # Query and load fasta sequences to runtime memory
        bed = BedTool(bed_str, from_string=True)
        sequences = []
        for line in open(bed.getfasta(genome_fa).seqfn).readlines():
            if not line.startswith('>'):
                sequences.append(line.strip().upper())

        # Group by splice event
        sequences = np.array_split(np.array(sequences), len(event_regions))
        vprint(f'| Loaded {len(sequences)} samples; {len(sequences[0])} sequences each')
        self.features = sequences
        
        # Define classes as sample
        labels = [event.sample for event in jc.events]
        self.labels = pd.factorize(labels)[0]
        
        # Define pytorch-style transforms
        self.transform = []
        if transform:
            self.transform = transform
        self.transform = Compose(self.transform + [ToOneHotEncoding()])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sequences = self.features[idx]
        label = self.labels[idx]
                
        sample = {'X': sequences, 'y': label}
        sample = self.transform(sample)
            
        return sample['X'], sample['y']
    
    
class PadSequence(object):
    '''
    Generate reverse complement of sample sequences.
    '''

    def __init__(self, seq_length):
        self.seq_length = seq_length
        pass
    
    def __call__(self, sample):
        sequences = sample['X']
        
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
        sample['X'] = new_seqs
        
        return sample  


class ReverseComplement(object):
    '''
    Generate reverse complement of sample sequences.
    '''

    def __init__(self):
        pass
    
    def __call__(self, sample):
        sequences = sample['X']
        
        new_seqs = []
        for seq in sequences:
            new_seqs.append(str(Seq(seq).reverse_complement()))
        sample['X'] = new_seqs
        
        return sample
        
class CropSequence(object):
    '''
    Generate reverse complement of sample sequence.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def __call__(self, sample):
        sequences = sample['X']
        
        new_seqs = []
        for seq in sequences:
            seq_len = len(seq)
            
            if seq_len != self.output_size:            
                start = np.random.randint(0, seq_len - self.output_size)
                end = start + self.output_size
                seq = seq[start:end]
                
            new_seqs.append(seq)
        sample['X'] = new_seqs
            
        return sample
    
class ToOneHotEncoding(object):
    '''
    Convert DNA sequence to one-hot encoded n x 4 array
    '''
    def __init__(self):
        self.base2index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    def __call__(self, sample):
        sequences = sample['X']
        new_seqs = []
        for seq in sequences:
            one_hot = np.zeros((len(seq), 4), dtype=np.float32)
            for i, base in enumerate(seq):
                if base in self.base2index:
                    one_hot[i, self.base2index[base]] = 1.
            new_seqs.append(np.transpose(one_hot))
            
        new_seqs = np.array(new_seqs)
#         vprint(new_seqs.shape)
#         new_seqs = np.transpose(new_seqs, (1, 0, 2))
            
        sample['X'] = new_seqs
        return sample
