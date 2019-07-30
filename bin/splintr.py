'''
Author : Clarence Mah

This module defines the custom PyTorch neural network Module and Dataset
for the neural network classifier structure using PyTorch for performing splice-event RBP classification. 
'''
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision.transforms import Compose


from pybedtools import BedTool
from Bio.Seq import Seq


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
                 fc_out):
        super().__init__()
        
        # Calculate appropriate padding to guarantee c1_out
        c1_padding = _calc_conv_pad(c1_in, c1_out, c1_kernel_w, c1_stride_w)
        
        # TODO make sure padding is > 0, < ?
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=c1_filter,
                      kernel_size=(1, c1_kernel_w),
                      stride=(1, c1_stride_w),
                      padding=(0, c1_padding)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)))
        
        c2_padding = _calc_conv_pad(c1_out/2, c2_out, c2_kernel_w, c2_stride_w)
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
        out = self.drop_out(out)
        out = self.fc1(out)
#         print('FC output: ', out.shape)
        out = self.output(out)
#         print('Final output: ', out.shape)
        return out


class SpliceEventDataset(Dataset):
    
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
        
        self.transform = []
        
        if transform:
            self.transform = transform

        self.transform = Compose(self.transform + [ToOneHotEncoding()])
        
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        sequences = [seq_group[idx] for seq_group in self.features]
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
    Convert ndarrays
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
        new_seqs = np.transpose(new_seqs, (1, 0, 2))
            
        sample['X'] = new_seqs
        return sample
    
def _calc_conv_pad(input_size, output_size, kernel_size, stride):
    '''
    Calculate appropriate padding to guarantee output size.
    '''
    return int((output_size * stride - input_size + kernel_size - stride) / 2)

