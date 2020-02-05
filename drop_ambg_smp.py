""" 
This script generates trainin of drug response regressor.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
from time import time
from pprint import pprint, pformat
from collections import OrderedDict
from glob import glob

import sklearn
import numpy as np
import pandas as pd

# File path
filepath = Path(__file__).resolve().parent

# Default settings
OUT_DIR = filepath / 'data'    
FILE_PATH = filepath / 'data/top_21.res_reg.cf_rnaseq.dd_dragon7.labled.parquet'
FILE_PATH_OUT = filepath / 'data/top_21.res_reg.cf_rnaseq.dd_dragon7.labled.ambg_drop.parquet'

def read_data_file(fpath, file_format='csv'):
    fpath = Path(fpath)
    if fpath.is_file():
        if file_format=='csv':
            df = pd.read_csv( fpath )
        elif file_format=='parquet':
            df = pd.read_parquet( fpath )
        else:
            raise ValueError('file format is not supported.')
    return df    

def bin_rsp(y, resp_thres=0.5):
    """ Binarize drug response values. """
    y = pd.Series( [0 if v>resp_thres else 1 for v in y.values] )
    return y
    
data = read_data_file( FILE_PATH, 'parquet' )

rsp = data[['CELL', 'DRUG', 'AUC']].copy()
rsp['y_true_cls'] = bin_rsp(rsp.AUC)

def find_ids_of_amb_samples(df):
    df = df.groupby(['CELL', 'DRUG']).agg({'y_true_cls': np.unique}).reset_index()
    df.insert(loc=1, column='source', value=[s.split('.')[0].lower() for s in df['CELL']]) # add 'source' column

    # print('\nSome samples contain ambiguous true labels (both 0 and 1).')
    # print(df.y_true_cls.value_counts()[:4])

    # print('\nThe unique types')
    # print(np.unique([str(type(x)) for x in df.y_true_cls]))

    print('\nCreate col indicating the number of unique responses per sample.')
    df['y_true_unq_vals'] = df.y_true_cls.map(lambda x: len(x) if type(x)==np.ndarray else 1)

    print('\nPrint bincount.')
    print(df.y_true_unq_vals.value_counts())

    print('\nExtract ambiguous samples.')
    df_amb = df[ df.y_true_unq_vals > 1 ].reset_index(drop=True)
    print(df_amb.shape)

    # Find indices of all the ambiguous samples 
    idx = np.array([False for i in range(rsp.shape[0])])
    for i in range(df_amb.shape[0]):
        cell_name = df_amb.loc[i,'CELL']
        drug_name = df_amb.loc[i,'DRUG']
        idx_ = (rsp.CELL==cell_name) & (rsp.DRUG==drug_name)
        idx = idx | idx_
    return idx

idx = find_ids_of_amb_samples(rsp)

print('\nTotal samples sue to ambiguous labels', sum(idx))
print('rsp    ', rsp.shape)
rsp_new = rsp[ ~idx ].reset_index(drop=True)
print('rsp_new', rsp_new.shape)
print('Percent of data to drop', (rsp.shape[0] - rsp_new.shape[0])/rsp.shape[0]*100 )

# Merge dataset with features
t0 = time()
data_new = pd.merge(rsp_new, data, on=['CELL', 'DRUG', 'AUC'], how='inner')
print('Merge time {:.2f} mins.'.format( (time() - t0)/60 ))

t0 = time()
data_new.to_parquet(FILE_PATH_OUT, index=False)
print('Save time {:.2f} mins.'.format( (time() - t0)/60 ))

print('data_new', data_new.shape)
print('Done.')





