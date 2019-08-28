'''
Author : Clarence Mah

This module defines the custom PyTorch neural network Module and Dataset
for the neural network classifier structure using PyTorch for performing splice-event RBP classification. 
'''
import splintr as sp
from splintr.splice import SpliceData
from splintr.util import vprint, calc_conv_pad, calc_conv_stride
from splintr.run import RunManager
import math 

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchsummary import summary

from pybedtools import BedTool
from Bio.Seq import Seq

from livelossplot import PlotLosses

from tqdm.autonotebook import tqdm
tqdm.pandas()
    

def load_datasets(feature_df, genome_fa, seq_size, augment_k=1):
    '''
    Loads pd.DataFrame
    '''
    # Randomize sample order
    rand_sample_i = np.random.choice(feature_df.shape[0], size=feature_df.shape[0], replace=False)

    # Determine dataset split size
    train_size, valid_size, test_size = [int(len(rand_sample_i) * s) for s in [0.8, 0.1, 0.1]]
    if np.sum([train_size, valid_size, test_size]) != len(rand_sample_i):
        train_size += 1
        
    vprint(f'Total samples: {len(rand_sample_i)}')
    vprint(f'Train: {train_size}, Validation: {valid_size}, Test: {test_size}')

    # Split into training, validation, and test
    vprint('Splitting data...')
    train_df = feature_df.iloc[rand_sample_i[:train_size]]
    valid_df = feature_df.iloc[rand_sample_i[train_size : train_size + valid_size]]
    test_df = feature_df.iloc[rand_sample_i[train_size + valid_size : train_size + valid_size + test_size]]
    datasets_df = [train_df, valid_df, test_df]

    # Additional parameters for loading data
    seq_length = seq_size
    genome_fa = '../data/hg19.fa'

    # Create PyTorch Dataset objects from each splice event k times
    vprint('Augmenting data...')
    datasets = []
    for df in tqdm(datasets_df, 'Datasets'):
        augmented_data = []
        for i in tqdm(range(augment_k), total=augment_k, leave=False):
            # Pad and crop transform
            tf1 = [sp.PadSequence(seq_length), sp.CropSequence(seq_length)]
            augmented_data.append(sp.SpliceEventDataset(feature_file=df,
                                                        genome_fa=genome_fa,
                                                        transform=tf1))

            # Pad and crop transform on reverse complement
            tf2 = [sp.PadSequence(seq_length), sp.CropSequence(seq_length), sp.ReverseComplement()]
            augmented_data.append(sp.SpliceEventDataset(feature_file=df,
                                                        genome_fa=genome_fa,
                                                        transform=tf2))
        augmented_data = torch.utils.data.ConcatDataset(augmented_data)
        datasets.append(augmented_data)

    # Convert categorical labels to numerical
    label_names = pd.factorize(feature_df['sample'])
    num_classes = int(max(label_names[0]) + 1)
    vprint(f'# of classes: {num_classes}')
    vprint(feature_df['sample'].value_counts())

    # Balance class sampling using weighted sampler
    vprint('Initializing samplers...')
    samplers = []
    for dataset in datasets:
        labels = [sample[1] for sample in dataset] # get label of each sample
        weights = 100. / pd.Series(labels).value_counts() # class weights
        weights = weights[labels].values 
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
        samplers.append(sampler)

    return datasets, samplers, label_names

def init_loaders(datasets, samplers, batch_size):
    # Create DataLoader for each dataset
    loaders = []
    for dataset, sampler in zip(datasets, samplers):
        loaders.append(DataLoader(dataset, batch_size=batch_size, sampler=sampler))
    
    return loaders
               
def fit(params, train_loader, valid_loader, num_epochs=10, label_names=None, log_dir=None):
    '''
    Train SplintrNet model.
    
    Parameters
    ----------
    params (dict) : parameters for training e.g. learning rate, batch size, etc.
        See fit source code and SplintrNet.init() for params used
    
    Returns
    -------
    net, valid_metric
    net (SplintrNet) : trained model
    valid_metric (float) : validation dataset metric as returned by RunManager.end_run()
    '''
    # Initialize model
    net = sp.SplintrNet(num_classes=6, params=params).cuda()
    summary(net, (4,4, 384))
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    manager = RunManager()
    
    # Perform training
    manager.begin_run(params, net, train_loader, valid_loader, label_names=label_names, log_dir=log_dir)
    net.cuda()
    for epoch in tqdm(range(num_epochs), total=num_epochs, leave=False):
        manager.begin_epoch()
        
        # Train on batch
        for batch in train_loader:
            seqs, labels = batch
            preds = net(seqs.cuda()) # pass batch
            loss = F.cross_entropy(preds, labels.cuda()) # calculate loss
            optimizer.zero_grad() # zero gradients
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            manager.track_train_loss(loss)
            manager.track_train_num_correct(preds, labels.cuda())
            
        # Check validation set
        with torch.no_grad():
            for data in valid_loader:
                seqs, labels = data
                preds = net(seqs.cuda())
                loss = F.cross_entropy(preds, labels.cuda())

                manager.track_valid_loss(loss)
                manager.track_valid_num_correct(preds, labels.cuda())
            
        # Tb logging
        manager.end_epoch()
        
    # Tb logging
    valid_metric = manager.end_run()
    return net, valid_metric
    
class SplintrNet(nn.Module):
    '''
    Input tensor size: [batch_size, in_channel, height, width]
    '''
    def __init__(self, num_classes, params):
        super().__init__()
        print(params)
        # Primary "k-mer" receptive field
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=params['filter1'],
                      kernel_size=(1, params['main_kernel']),
                      stride=(1, 3),
                      padding=(0, calc_conv_pad(384, 128, params['main_kernel'], 3))),
            nn.ReLU())
        
        # Dynamic receptive field depth; min 1
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=params['filter1'],
                                             out_channels=params['filter2'],
                                             kernel_size=(1, 3),
                                             stride=(1, 1),
                                             padding=(0, calc_conv_pad(128, 128, 3, 1))),
                                   nn.ReLU())
        
        # Define filter block
        mini_module = [nn.Conv2d(in_channels=params['filter2'],
                                 out_channels=params['filter2'],
                                 kernel_size=(1, 3),
                                 stride=(1, 1),
                                 padding=(0, calc_conv_pad(128, 128, 3, 1))),
                       nn.ReLU()]
        
        # Compounded filters
        if params['receptive_layers'] > 0:
            mini_module_list = np.tile(mini_module, params['receptive_layers']).tolist()
            for i, module in enumerate(mini_module_list):
                self.conv2.add_module(str(i+2), module)

        # Max pool layer
        self.conv2.add_module('maxpool', nn.MaxPool2d(kernel_size=(1, 2)))
        
        self.drop_out = nn.Dropout(params['dropout'])
        
        fc_in = int(params['filter2']*4*64)
        self.fc1 = nn.Linear(in_features=fc_in, out_features=params['fc_out'])
        self.output = nn.Sequential(
            nn.Linear(in_features=params['fc_out'], out_features=num_classes),
            nn.Softmax())

        
    def forward(self, x):
#         print('Input: ', x.shape)
        out = self.conv1(x)
#         print('CNN1 output: ', out.shape)
        out = self.conv2(out)
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
