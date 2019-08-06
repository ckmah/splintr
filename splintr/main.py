#!/usr/bin/env python
import os
import argparse

import pandas as pd
import numpy as np

from shutil import copyfile
from tqdm.autonotebook import tqdm

import multiprocessing
from multiprocessing import Pool

def remove_duplicate_events(df):
    duplicated = df.iloc[:, 5:-14].duplicated()
    df = df.loc[~duplicated]
    return df

parser = argparse.ArgumentParser(description='Splintr neural network classifier.')
subparsers = parser.add_subparsers(dest='command')

prep_parser = subparsers.add_parser('prep')
prep_parser.add_argument(dest='encore_rmats_dir', type=str, help='Input dir for ENCORE RMATS alt. splicing files')
prep_parser.add_argument(dest='output_dir', type=str, help='Output dir for aggregated training files')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('arg1', type=str, help='idk yet')

args = parser.parse_args()

# if args.command:
    
# rmats_dir = args.encore_rmats_dir
# output_dir = args.output_dir

# # Iterate over event -> cell line -> RBP
# def generate_features(event):
#     cell_lines = os.listdir(f'{rmats_dir}/{event}')

#     # cell line
#     event_all_cell_dataset = []
#     for cell_line in tqdm(cell_lines, position=1, desc='Cell lines', leave=False):
#         jc_files = os.listdir(f'{rmats_dir}/{event}/{cell_line}')

#         # sample
#         cell_datasets = []
#         for jc_file in tqdm(jc_files, position=2, desc='Samples', leave=False):
#             jc_filepath = f'{rmats_dir}/{event}/{cell_line}/{jc_file}'
#             sample_name = jc_file.split('.')[0] # parse sample name

#             # Load junction counts data
#             jc = pd.read_csv(jc_filepath, sep='\t')
#             jc['event'] = event

#             # Set aside non-alternatively spliced events as control
#             bg_jc = jc.loc[jc['FDR'] > 0.1]
#             bg_jc = bg_jc.iloc[np.random.randint(bg_jc.shape[0], size=1)]
#             bg_jc['sample'] = 'bg'

#             # Alternatively spliced events
#             jc = jc.loc[jc['FDR'] < 0.1]
#             jc['sample'] = sample_name
#             jc = pd.concat([bg_jc, jc])

#             cell_datasets.append(jc)

#         # Combine all events for event type
#         cell_datasets = pd.concat(cell_datasets, ignore_index=True)

#         # Remove duplicates
#         cell_datasets = remove_duplicate_events(cell_datasets)
#         event_all_cell_dataset.append(cell_datasets)

#         # Write to file
#         file_prefix = f'{output_dir}/{cell_line}_{event}'
#         cell_datasets.to_csv(f'{file_prefix}.txt', sep='\t', index=False)

#     # Combine all events for given event type across all cell lines
#     event_all_cell_dataset = pd.concat(event_all_cell_dataset, ignore_index=True)

#     # remove duplicates
#     event_all_cell_dataset = remove_duplicate_events(event_all_cell_dataset)
#     event_all_cell_dataset.to_csv(f'{output_dir}/{event}.txt', sep='\t', index=False)

# events = os.listdir(rmats_dir)
# p = Pool(4)
# list(tqdm(p.imap_unordered(generate_features, events), total=len(events)))




