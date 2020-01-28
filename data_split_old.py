from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import platform
import argparse
from pathlib import Path
from pprint import pprint, pformat

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

SEED = 0

# DATAPATH = '../top_21.res_reg.cf_rnaseq.dd_dragon7.labeled.hdf5'
DATAPATH = './top_21.res_reg.cf_rnaseq.dd_dragon7.labled.parquet'
outdir = Path('./')

# File path
filepath = Path(__file__).resolve().parent


def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate data partitions.")
    args = None

    # Parse args
    args = parser.parse_args(args)
    return args



# -----------------------------------------------------------
# Code below comes from cv_splitter.py
# -----------------------------------------------------------
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def cv_splitter(cv_method: str='simple', cv_folds: int=1, test_size: float=0.2,
                mltype: str='reg', shuffle: bool=False, random_state=None):
    """ Creates a cross-validation splitter.
    Args:
        cv_method: 'simple', 'stratify' (only for classification), 'groups' (only for regression)
        cv_folds: number of cv folds
        test_size: fraction of test set size (used only if cv_folds=1)
        mltype: 'reg', 'cls'
    """
    # Classification
    if mltype == 'cls':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
            
        elif cv_method == 'stratify':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)

    # Regression
    elif mltype == 'reg':
        # Regression
        if cv_method == 'group':
            if cv_folds == 1:
                cv = GroupShuffleSplit(n_splits=cv_folds, random_state=random_state)
            else:
                cv = GroupKFold(n_splits=cv_folds)
            
        elif cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
    return cv


def plot_ytr_yvl_dist(ytr, yvl, title=None, outpath='.'):
    """ Plot distributions of response data of train and val sets. """
    fig, ax = plt.subplots()
    plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
    plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
    if title is None: title = ''
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    if outpath is None:
        plt.savefig(Path(outpath)/'ytr_yvl_dist.png', bbox_inches='tight')
    else:
        plt.savefig(outpath, bbox_inches='tight')
# -----------------------------------------------------------


        
def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))

            
def plot_hist(x, var_name, fit=None, bins=100, path='hist.png'):
    """ Plot hist of a 1-D array x. """
    if fit is not None:
        (mu, sigma) = stats.norm.fit(x)
        fit = stats.norm
        label = f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'
    else:
        label = ''
    
    alpha = 0.6
    fig, ax = plt.subplots()
#     sns.distplot(x, bins=bins, kde=True, fit=fit, 
#                  hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'},
#                  kde_kws={'linewidth': 2, 'alpha': alpha, 'color': 'k'},
#                  fit_kws={'linewidth': 2, 'alpha': alpha, 'color': 'r',
#                           'label': label})
    sns.distplot(x, bins=bins, kde=False, fit=fit, 
                 hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'})
    plt.grid(True)
    plt.legend()
    plt.title(var_name + ' hist')
    plt.savefig(path, bbox_inches='tight')
    
            
def make_split(args):

    # Load data
    data = pd.read_parquet( DATAPATH ) 

    n_runs = 100
    for seed in range(n_runs):
        digits = len(str(n_runs))
        seed_str = f"{seed}".zfill(digits)
        output = '1fold_s' + seed_str 

        te_method = 'simple'
        cv_method = 'simple'
        te_size = 0.1
        vl_size = 0.1

        # Features 
        cell_fea = 'GE'
        drug_fea = 'DD'
        
        # Other params
        n_jobs = 8

        # Hard split
        grp_by_col = None

        # TODO: this need to be improved
        mltype = 'reg'  # required for the splits (stratify in case of classification)
        
        
        # -----------------------------------------------
        #       Train-test split
        # -----------------------------------------------
        np.random.seed(seed)
        idx_vec = np.random.permutation(data.shape[0])

        if te_method is not None:
            te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
                                      mltype=mltype, shuffle=False, random_state=seed)

            te_grp = meta[grp_by_col].values[idx_vec] if te_method=='group' else None
            if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
       
            # Split train/test
            tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
            tr_id = idx_vec[tr_id] # adjust the indices!
            te_id = idx_vec[te_id] # adjust the indices!

            pd.Series(tr_id).to_csv(outdir/f'{output}_tr_id.csv', index=False, header=[0])
            # pd.Series(te_id).to_csv(outdir/f'{output}_te_id.csv', index=False, header=[0])
            pd.Series(te_id).to_csv(outdir/f'{output}_vl_id.csv', index=False, header=[0])
            
            
            # Confirm that group splits are correct
            if te_method=='group' and grp_by_col is not None:
                tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
                te_grp_unq = set(meta.loc[te_id, grp_by_col])
                lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tA few intersections : {list(tr_grp_unq.intersection(te_grp_unq))[:3]}.')


def main(args):
    args = parse_args(args)
    args = vars(args)
    # run(args)
    make_split(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])



