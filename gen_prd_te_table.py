import os
import sys
from pathlib import Path
from time import time
from glob import glob

# import sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns

filepath = Path(__file__).resolve().parent

# As input arg provide the path do the dir that contains the 100 runs
# args = sys.argv[1:]
# trn_path = Path(args[0])


def reorg_cols(df, col_first:str):
    """
    Args:
        col_first : col name to put first
    """
    cols = df.columns.tolist()
    cols.remove(col_first)
    return df[[col_first] + cols]
    

def agg_preds_from_cls_runs(runs_dirs, phase='_te.csv', verbose=False):
    """ Aggregate predictions bootstraped ML trainings. """
    prd = []
    for i, dir_name in enumerate(runs_dirs):
        if '_tr.csv' in phase:
            prd_ = pd.read_csv(dir_name/'preds_tr.csv')
        elif '_vl.csv' in phase:
            prd_ = pd.read_csv(dir_name/'preds_vl.csv')
        elif '_te.csv' in phase:
            prd_ = pd.read_csv(dir_name/'preds_te.csv')
        
        # prd_te_['err'] = abs(prd_te_['y_true'] - prd_te_['y_pred'])      # add col 'err'
        prd_['run'] = str(dir_name).split(os.sep)[-1].split('_')[-1]  # add col 'run' identifier
        prd.append(prd_)  # append run data

        if verbose:
            if i%20==0:
                print(f'Processing {dir_name}')
            
    # Aggregate to df
    prd = pd.concat(prd, axis=0)
    
    # Reorganize cols
    prd = reorg_cols(prd, col_first='run').sort_values('run').reset_index(drop=True).reset_index().rename(columns={'index': 'idx'})
    return prd


def update_prd_table(prd_df, cancer_types=None):
    """ ... """
    # Add SOURCE column
    if 'source' not in [str(i).lower() for i in prd_df.columns.to_list()]:
        prd_df.insert(loc=2, column='SOURCE', value=[s.split('.')[0].lower() for s in prd_df['CELL']])

    # Add CTYPE column
    prd_df = pd.merge(prd_df, cancer_types, on='CELL')
    prd_df = reorg_cols(prd_df, col_first='CTYPE')

    # Rename
    prd_df = prd_df.rename(columns={'y_true': 'y_true_cls', 'y_pred': 'y_pred_prob'})

    # Retain specific columns
    cols = ['idx', 'run', 'SOURCE', 'CTYPE', 'CELL', 'DRUG', 'R2fit', 'AUC', 'y_true_cls', 'y_pred_prob']
    prd_df = prd_df[cols]

    # Add col of pred labels
    prd_df['y_pred_cls'] = prd_df.y_pred_prob.map(lambda x: 0 if x<0.5 else 1)

    # The highest error is 0.5 while the lowest is 0.
    # This value is proportional to the square root of Brier score.
    prd_te_all['prob_err'] = abs(prd_te_all.y_true_cls - prd_te_all.y_pred_prob)

    # Bin AUC values 
    bins = np.arange(0, 1.1, 0.1).tolist()
    prd_df['AUC_bin'] = pd.cut(prd_df.AUC, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise')

    # Add col that cetegorizes the preds
    prd_df['prd_cat'] = None
    prd_df.prd_cat[ (prd_df.y_true_cls==1) & (prd_df.y_pred_cls==1) ] = 'TP'
    prd_df.prd_cat[ (prd_df.y_true_cls==0) & (prd_df.y_pred_cls==0) ] = 'TN'
    prd_df.prd_cat[ (prd_df.y_true_cls==1) & (prd_df.y_pred_cls==0) ] = 'FN'
    prd_df.prd_cat[ (prd_df.y_true_cls==0) & (prd_df.y_pred_cls==1) ] = 'FP'

    # Add cols
    prd_df['TP'] = prd_df.prd_cat=='TP'
    prd_df['TN'] = prd_df.prd_cat=='TN'
    prd_df['FP'] = prd_df.prd_cat=='FP'
    prd_df['FN'] = prd_df.prd_cat=='FN'

    return prd_df


def run(args):
    # Cancer types
    cancer_types = pd.read_csv(filepath/'data/combined_cancer_types', sep='\t', names=['CELL', 'CTYPE'])

    # Concat preds from all runs
    runs_dirs = [Path(p) for p in glob(str(trn_path/'run_*'))]
    prd_te_all = agg_preds_from_cls_runs(runs_dirs, phase='_te.csv')
    prd_te_all = update_prd_table(prd_te_all, cancer_types=cancer_types)

    # Save aggregated master table
    prd_te_all.to_csv(trn_path/'prd_te_all.csv', index=False)

    # Confusion Matrix
    conf = confusion_matrix( prd_te_all.y_true_cls, prd_te_all.y_pred_cls, normalize=None )
    conf_plot = ConfusionMatrixDisplay(conf, display_labels=['NoResp', 'Resp'])
    conf_plot.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation=None, values_format='d')
    plt.show()
    plt.savefig(trn_path/'conf_mat.png', dpi=100)

    # Confusion Matrix (normalized)
    conf = confusion_matrix( prd_te_all.y_true_cls, prd_te_all.y_pred_cls, normalize='all' )
    conf_plot = ConfusionMatrixDisplay(conf, display_labels=['NoResp', 'Resp'])
    conf_plot.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation=None, values_format='.2f')
    plt.savefig(trn_path/'conf_mat_norm.png', dpi=100)


def main(args):
    # args = parse_args(args)
    # args = vars(args)
    run(args)
    # print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])
