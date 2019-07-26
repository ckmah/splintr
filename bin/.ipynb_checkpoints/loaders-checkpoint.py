'''
Taken from @Avsecz/batch_generator.py on GitHub.
Source: https://gist.github.com/Avsecz/464980fb748b10dc71530e1e9e778f2e
'''
from torch.utils.data import Dataset, DataLoader
from genomelake.extractors import FastaExtractor
import pybedtools
from pybedtools import BedTool
import linecache 
import numpy as np
import pandas as pd

class BedToolLinecache(BedTool):
    """Fast BedTool accessor by Ziga Avsec
    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))

class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    def __init__(self, 
		 intervals_file, 
		 fasta_file,
                 target_file=None, 
		 use_linecache=True):

        # intervals
        if use_linecache:
            self.bt = BedToolLinecache(intervals_file)
        else:
            self.bt = BedTool(intervals_file)
        self.fasta_file = fasta_file
        self.fasta_extractor = None

        # Targets
        if target_file is not None:
            self.targets = pd.read_csv(target_file, header=None)
        else:
            self.targets = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
	    # initialize the fasta extractor on each worker separately
            self.fasta_extractor = FastaExtractor(self.fasta_file)
        interval = self.bt[idx]

        if self.targets is not None:
            target = self.targets.iloc[idx].values[0]
        else:
            target = None

        # Run the fasta extractor
        seq = self.fasta_extractor([interval])
        # Squeeze is required because we are returning individual samples
	
        return seq, target
    