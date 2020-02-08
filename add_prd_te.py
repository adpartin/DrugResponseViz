import os
import sys
from pathlib import Path
from time import time
from glob import glob

# import sklearn
import numpy as np
import pandas as pd

filepath = Path(__file__).resolve().parent

args = sys.argv[1:]
trn_path = Path(args[0])


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

# Concat preds from all runs
runs_dirs = [Path(p) for p in glob(str(trn_path/'run_*'))]
prd_te_all = agg_preds_from_cls_runs(runs_dirs, phase='_te.csv')
prd_te_all.insert(loc=2, column='source', value=[s.split('.')[0].lower() for s in prd_te_all['CELL']])

# Cancer types
cancer_types = pd.read_csv(filepath/'data/combined_cancer_types', sep='\t', names=['CELL', 'CTYPE'])

# Add CTYPE columns
prd_te_all = pd.merge(prd_te_all, cancer_types, on='CELL')
prd_te_all = reorg_cols(prd_te_all, col_first='CTYPE')

prd_te_all = prd_te_all.rename(columns={'y_true': 'y_true_cls'})
prd_te_all['y_pred_cls'] = prd_te_all.y_pred.map(lambda x: 0 if x<0.5 else 1)

bins = np.arange(0, 1.1, 0.1).tolist()
prd_te_all['AUC_bin'] = pd.cut(prd_te_all.AUC, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise')

prd_te_all['prd_cat'] = None
prd_te_all.prd_cat[ (prd_te_all.y_true_cls==1) & (prd_te_all.y_pred_cls==1) ] = 'TP'
prd_te_all.prd_cat[ (prd_te_all.y_true_cls==0) & (prd_te_all.y_pred_cls==0) ] = 'TN'
prd_te_all.prd_cat[ (prd_te_all.y_true_cls==1) & (prd_te_all.y_pred_cls==0) ] = 'FN'
prd_te_all.prd_cat[ (prd_te_all.y_true_cls==0) & (prd_te_all.y_pred_cls==1) ] = 'FP'

# Save aggregated master table
prd_te_all.to_csv(trn_path/'prd_te_all.csv', index=False)


