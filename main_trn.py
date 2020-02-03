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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.externals import joblib
from math import sqrt
t0 = time()

# File path
filepath = Path(__file__).resolve().parent

# Utils
from classlogger import Logger
import ml_models
    
# Default settings
OUT_DIR = filepath / 'out'    
FILE_PATH = filepath / 'data/top_21.res_reg.cf_rnaseq.dd_dragon7.labled.parquet'
SPLITS_DIR = filepath / 'splits_old'

        
def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate learning curves.")

    # Input data
    parser.add_argument('-fp', '--filepath', default=FILE_PATH, type=str, help='Full path to data (default: None).')

    # Path to splits
    parser.add_argument('-sp', '--splitpath', default=SPLITS_DIR, type=str, help='Full path to data splits (default: ./splits).')
    parser.add_argument('--n_splits', default=None, type=int, help='Use a subset of splits (default: None).')

    # Global outdir
    parser.add_argument('-gout', '--global_outdir', default=OUT_DIR, type=str, help='Gloabl outdir. (default: out).')
    parser.add_argument('-rout', '--run_outdir', default=None, type=str, help='Run outdir. This is the for the specific run (default: None).')

    # Select target to predict
    parser.add_argument('-t', '--target_name', default='AUC', type=str, choices=['AUC'], help='Name of target variable (default: AUC).')

    # Select feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell line features (default: GE).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: DD).')

    # Data split methods
    parser.add_argument('-cvf', '--cv_folds', default=1, type=str, help='Number cross-val folds (default: 1).')
    parser.add_argument('-cvf_arr', '--cv_folds_arr', nargs='+', type=int, default=None, help='The specific folds in the cross-val run (default: None).')
    
    # ML models
    parser.add_argument('-frm', '--framework', default='lightgbm', type=str, choices=['keras', 'lightgbm', 'sklearn'], help='ML framework (default: lightgbm).')
    parser.add_argument('-ml', '--model_name', default='lgb_cls', type=str,
                        choices=['lgb_reg', 'rf_reg', 'nn_reg', 'lgb_cls'], help='ML model (default: lgb_cls).')

    # LightGBM params
    parser.add_argument('--gbm_leaves', default=31, type=int, help='Maximum tree leaves for base learners (default: 31).')
    parser.add_argument('--gbm_lr', default=0.1, type=float, help='Boosting learning rate (default: 0.1).')
    parser.add_argument('--gbm_max_depth', default=-1, type=int, help='Maximum tree depth for base learners (default: -1).')
    parser.add_argument('--gbm_trees', default=100, type=int, help='Number of trees (default: 100).')
    
    # Random Forest params
    parser.add_argument('--rf_trees', default=100, type=int, help='Number of trees (default: 100).')   
    
    # NN hyper_params
    parser.add_argument('-ep', '--epochs', default=200, type=int, help='Number of epochs (default: 200).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    parser.add_argument('-sc', '--scaler', default='stnd', type=str, choices=['stnd', 'minmax', 'rbst'],
            help='Feature normalization method (stnd, minmax, rbst) (default: stnd).')
    parser.add_argument('--batchnorm', action='store_true', help='Whether to use batch normalization (default: False).')
    # parser.add_argument('--residual', action='store_true', help='Whether to use residual conncetion (default: False).')
    # parser.add_argument('--initializer', default='he', type=str, choices=['he', 'glorot'], help='Keras initializer name (default: he).')

    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer name (default: sgd).')
    parser.add_argument('--lr', default='0.0001', type=float, help='Learning rate of adaptive optimizers (default: 0.001).')

    parser.add_argument('--clr_mode', default=None, type=str, choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1).')
    parser.add_argument('--clr_base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--clr_max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--clr_gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    # Other
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 8.')
    parser.add_argument('--seed', default=0, type=int, help='Default: 0.')

    # Parse args
    # args = parser.parse_args(args)
    args, other_args = parser.parse_known_args(args)
    return args
        

def verify_dirpath(dirpath):
    """ Verify the dirpath exists and contain the dataset. """
    if dirpath is None:
        sys.exit('Program terminated. You must specify a path to a data via the input argument -dp.')

    dirpath = Path(dirpath)
    assert dirpath.exists(), 'The specified dirpath was not found: {dirpath}.'
    return dirpath
    
    
def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))
    

def get_file(fpath):
    return pd.read_csv(fpath, header=None).squeeze().values if fpath.is_file() else None


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
    
    
def scale_fea(xdata, scaler_name='stnd', dtype=np.float32):
    """ Returns the scaled dataframe of features. """
    if scaler_name is not None:
        if scaler_name == 'stnd':
            scaler = StandardScaler()
        elif scaler_name == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_name == 'rbst':
            scaler = RobustScaler()
    
    cols = xdata.columns
    return pd.DataFrame( scaler.fit_transform(xdata), columns=cols, dtype=dtype )    
    

def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    
            
def get_data_by_id(idx, X, Y, meta=None):
    """ Returns a tuple of (features (x), target (y), metadata (m))
    for an input array of indices (idx). """
    x_data = X.iloc[idx, :].reset_index(drop=True)
    y_data = np.squeeze(Y.iloc[idx, :]).reset_index(drop=True)
    if meta is not None:
        m_data = meta.iloc[idx, :].reset_index(drop=True)
    else:
        meta = None
    return x_data, y_data, m_data


def trn_lgbm_model(model, xtr, ytr, fit_kwargs, eval_set=None):
    """ Train and save LigthGBM model. """
    # Fit params
    fit_kwargs = fit_kwargs
    # fit_kwargs = self.fit_kwargs.copy()
    fit_kwargs['eval_set'] = eval_set
    fit_kwargs['early_stopping_rounds'] = 10

    # Train and save model
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    runtime = (time() - t0)/60

    # Remove key (we'll dump this dict so we don't need to print all the eval set)
    fit_kwargs.pop('eval_set', None)

    # joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
    return model, runtime


def trn_sklearn_model(model, xtr, ytr, fit_kwargs, eval_set=None):
    """ Train and save sklearn model. """
    # Fit params
    fit_kwargs = fit_kwargs
    # fit_kwargs = self.fit_kwargs.copy()

    # Train and save model
    t0 = time()
    model.fit(xtr, ytr, **fit_kwargs)
    runtime = (time() - t0)/60
    # joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
    return model, runtime


def create_trn_outdir(fold, tr_sz):
    trn_outdir = outdir / ('cv'+str(fold) + '_sz'+str(tr_sz))
    os.makedirs(trn_outdir, exist_ok=True)
    return trn_outdir
    

def calc_preds(model, x, y, mltype):
    """ Calc predictions. """
    if mltype == 'cls': 
        def get_pred_fn(model):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba
            if hasattr(model, 'predict'):
                return model.predict

        pred_fn = get_pred_fn(model)
        if (y.ndim > 1) and (y.shape[1] > 1):
            y_pred = pred_fn(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(ydata, axis=1)
        else:
            y_pred = pred_fn(x)
            y_true = y
            
    elif mltype == 'reg':
        y_pred = np.squeeze(model.predict(x))
        y_true = np.squeeze(y)

    return y_pred, y_true


def dump_preds(y_true, y_pred, meta=None, outpath='./preds.csv'):
    """ Dump prediction and true values, with optional with metadata. """
    y_true = pd.Series(y_true, name='y_true')
    y_pred = pd.Series(y_pred, name='y_pred')
    if meta is not None:
        preds = meta.copy()
        preds.insert(loc=3, column='y_true', value=y_true.values)
        preds.insert(loc=4, column='y_pred', value=y_pred.values)
    else:
        preds = pd.concat([y_true, y_pred], axis=1)
    preds.to_csv(Path(outpath), index=False)


def calc_scores(y_true, y_pred, mltype, metrics=None):
    """ Create dict of scores.
    Args:
        metrics : TODO allow to pass a string of metrics
    """
    scores = {}

    if mltype == 'cls':    
        # Metric that accept probabilities
        scores['brier'] = sklearn.metrics.brier_score_loss(y_true, y_pred, sample_weight=None, pos_label=1)
        scores['auc_roc'] = sklearn.metrics.roc_auc_score(y_true, y_pred)

        # Metric that don't accept probabilities
        y_pred_ = [1 if v>0.5 else 0 for v in y_pred]
        scores['mcc'] = sklearn.metrics.matthews_corrcoef(y_true, y_pred_, sample_weight=None)
        scores['f1_score'] = sklearn.metrics.f1_score(y_true, y_pred_, average='micro')
        scores['acc_blnc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred_)

    elif mltype == 'reg':
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        # scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores['mse'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores['rmse'] = sqrt( sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred) )
        # scores['auroc_reg'] = reg_auroc(y_true=y_true, y_pred=y_pred)
        
    scores['y_avg_true'] = np.mean(y_true)
    scores['y_avg_pred'] = np.mean(y_pred)
    return scores


def scores_to_df(scores_all):
    """ Dict to df. """
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['run'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['run'], columns=['metric'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


def bin_rsp(y, resp_thres=0.5):
    """ Binarize drug response values. """
    y = pd.Series( [0 if v>resp_thres else 1 for v in y.values] )
    return y


def get_model_kwargs(args):
    """ Get ML model init and fit agrs. """
    if args['framework'] == 'lightgbm':
        model_init_kwargs = { 'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
                              'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
                              'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
        model_fit_kwargs = {'verbose': False}

    elif args['framework'] == 'sklearn':
        model_init_kwargs = { 'n_estimators': args['rf_trees'], 'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
        model_fit_kwargs = {}

    elif args['framework'] == 'keras':
        model_init_kwargs = { 'input_dim': data.shape[1], 'dr_rate': args['dr_rate'],
                              'opt_name': args['opt'], 'lr': args['lr'], 'batchnorm': args['batchnorm']}
        model_fit_kwargs = { 'batch_size': args['batch_size'], 'epochs': args['epochs'], 'verbose': 1 }        
    
    return model_init_kwargs, model_fit_kwargs


def run(args):
    # Global outdir
    gout = Path(args['global_outdir'])
    os.makedirs(gout, exist_ok=True)

    # Create logger
    lg = Logger(gout/'logfile.log')
    lg.logger.info(f'File path: {filepath}')
    lg.logger.info(f'\n{pformat(args)}')

    # dirpath = verify_dirpath(args['dirpath'])
    data = read_data_file( args['filepath'], 'parquet' )
    print('data.shape', data.shape)

    # Get features (x), target (y), and meta
    fea_list = args['cell_fea'] + args['drug_fea']
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep='_')
    meta = data.drop(columns=xdata.columns)
    ydata = meta[[ args['target_name'] ]]

    # ML type ('reg' or 'cls')
    if 'reg' in args['model_name']:
        mltype = 'reg'
    elif 'cls' in args['model_name']:
        mltype = 'cls'
    else:
        raise ValueError("model_name must contain 'reg' or 'cls'.")

    # Find out which metadata field was used for hard split (cell, drug, or none)
    # f = [f for f in dirpath.glob('*args.txt')][0]
    # with open(f) as f: lines = f.readlines()
    # def find_arg(lines, arg):
    #     return [l.split(':')[-1].strip() for l in lines if arg+':' in l][0]
    # args['split_on'] = find_arg(lines, 'split_on').lower()
    # args['split_seed'] = find_arg(lines, 'seed').lower()

    def get_unq_split_ids(all_splits_path):
        """ List containing the full path of each split. """
        unq = [all_splits_path[i].split(os.sep)[-1].split('_')[1] for i, p in enumerate(all_splits_path)]
        # unq = []
        # for i, p in enumerate(all_splits_path): 
        #     sp = all_splits_path[i].split(os.sep)[-1].split('_')[1]
        # unq.append(sp)
        unq = np.unique(unq)
        return unq

    all_splits_path = glob(str(Path(args['splitpath'])/'1fold_*_id.csv'))
    unq_split_ids = get_unq_split_ids(all_splits_path)
    run_times = []

    # Append scores (dicts)
    tr_scores_all = []
    vl_scores_all = []
    te_scores_all = []

    # Sample size at each run
    smp_sz = []
    file_smp_sz = open(gout/'sample_sz', 'w')
    file_smp_sz.write('run\ttr_sz\tvl_sz\tte_sz\n')

    # Iterate over splits
    n_splits = args['n_splits']
    for i, split_id in enumerate(unq_split_ids[:n_splits]):
        # print(f'Split {split_id}')

        # Get indices for the split
        aa = [p for p in all_splits_path if f'1fold_{split_id}' in p]
        if len(aa) < 2:
            print(f'The split {s} contains only one file.')
            continue
        for id_file in aa:
            if 'tr_id' in id_file:
                tr_id = read_data_file( id_file )
            elif 'vl_id' in id_file:
                # vl_id = read_data_file( id_file )
                te_id = read_data_file( id_file )

        # Define run outdir
        rout = gout/f'run_{split_id}'
        os.makedirs(rout, exist_ok=True)

        # Scaling
        # xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])  # scale features

        # Get training and val data
        # Extract Train set T, Validation set V, and Test set E
        # tr_id.iloc[:, fold].dropna().values.astype(int).tolist()
        tr_id = tr_id.iloc[:,0].values.astype(int).tolist()
        # vl_id = vl_id.iloc[:,0].values.astype(int).tolist()
        te_id = te_id.iloc[:,0].values.astype(int).tolist()
        xtr, ytr, mtr = get_data_by_id(tr_id, xdata, ydata, meta) # samples from xtr are sequentially sampled for TRAIN
        # xvl, yvl, mvl = get_data_by_id(vl_id, xdata, ydata, meta) # fixed set of VAL samples for the current CV split
        xte, yte, mte = get_data_by_id(te_id, xdata, ydata, meta) # fixed set of TEST samples for the current CV split

        # Extract val data
        from sklearn.model_selection import train_test_split
        id_arr = np.arange(len(xtr))
        tr_, vl_ = train_test_split(id_arr, test_size=0.1)
        xvl = xtr.iloc[vl_,:].reset_index(drop=True)
        xtr = xtr.iloc[tr_,:].reset_index(drop=True)
        mvl = mtr.iloc[vl_,:].reset_index(drop=True)
        mtr = mtr.iloc[tr_,:].reset_index(drop=True)
        yvl = ytr.iloc[vl_].reset_index(drop=True)
        ytr = ytr.iloc[tr_].reset_index(drop=True)

        def drop_samples(x_df, y_df, m_df, items_to_drop, drop_by='CELL'):
            id_drop = m_df[drop_by].isin( items_to_drop )
            x_df = x_df[~id_drop].reset_index(drop=True)
            y_df = y_df[~id_drop].reset_index(drop=True)
            m_df = m_df[~id_drop].reset_index(drop=True)
            return x_df, y_df, m_df

        # Dump cell lines
        cell_to_drop_fname = 'cell_list_tmp'
        cell_to_drop_fpath = filepath / cell_to_drop_fname
        if cell_to_drop_fpath.exists():
            with open(cell_to_drop_fpath, 'r') as f:
                cells_to_drop = [line.rstrip() for line in f]
                xtr, ytr, mtr = drop_samples(x_df=xtr, y_df=ytr, m_df=mtr, items_to_drop=cells_to_drop)
                xvl, yvl, mvl = drop_samples(x_df=xvl, y_df=yvl, m_df=mvl, items_to_drop=cells_to_drop)
                xte, yte, mte = drop_samples(x_df=xte, y_df=yte, m_df=mte, items_to_drop=cells_to_drop)

        line = 's{}\t{}\t{}\t{}\n'.format(split_id, xtr.shape[0], xvl.shape[0], xte.shape[0])
        file_smp_sz.write(line)

        # Adjust the responses
        if mltype=='cls':
            ytr = bin_rsp(ytr, resp_thres=0.5)
            yvl = bin_rsp(yvl, resp_thres=0.5)
            yte = bin_rsp(yte, resp_thres=0.5)

        # Define ML model
        if 'lgb' in args['model_name']:
            args['framework'] = 'lightgbm'
        elif args['model_name'] == 'rf_reg':
            args['framework'] = 'sklearn'
        elif 'nn_' in args['model_name']:
            args['framework'] = 'keras'

        model_init_kwargs, model_fit_kwargs = get_model_kwargs(args)

        # Get the estimator
        estimator = ml_models.get_model(args['model_name'], init_kwargs=model_init_kwargs)
        model = estimator.model
        
        # Train
        eval_set = (xvl, yvl)
        # eval_set = None
        if args['framework']=='lightgbm':
            model, runtime = trn_lgbm_model(model=model, xtr=xtr, ytr=ytr,
                                            eval_set=eval_set, fit_kwargs=model_fit_kwargs)
        elif args['framework']=='sklearn':
            model, runtime = trn_sklearn_model(model=model, xtr_sub=xtr, ytr_sub=ytr,
                                               eval_set=None, fit_kwargs=model_fit_kwargs)
        elif args['framework']=='keras':
            model, runtime = trn_keras_model(model=model, xtr_sub=xtr, ytr_sub=ytr,
                                             eval_set=eval_set)
        elif args['framework']=='pytorch':
            pass
        else:
            raise ValueError(f'Framework {framework} is not yet supported.')
            
        if model is None:
            continue # sometimes keras fails to train a model (evaluates to nan)

        # Append runtime
        run_times.append(runtime)
        
        # Dump model
        if args['save_model']:
            joblib.dump(model, filename = rout / ('model.'+args['model_name']+'.pkl') )

        # Calc preds and scores
        # ... training set
        y_pred, y_true = calc_preds(model, x=xtr, y=ytr, mltype=mltype)
        tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=mtr, outpath=rout/'preds_tr.csv')
        # ... val set
        y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=mltype)
        vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=mvl, outpath=rout/'preds_vl.csv')
        # ... test set
        y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype=mltype)
        te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=mltype, metrics=None)
        dump_preds(y_true, y_pred, meta=mte, outpath=rout/'preds_te.csv')

        # Add metadata
        tr_scores['run'] = split_id
        vl_scores['run'] = split_id
        te_scores['run'] = split_id

        # Append scores (dicts)
        tr_scores_all.append(tr_scores)
        vl_scores_all.append(vl_scores)
        te_scores_all.append(te_scores)

        # Free space
        del xtr, ytr, mtr, xvl, yvl, mvl, xte, yte, mte, tr_, vl_

        if i%10 == 0:
            print(f'Finished {split_id}')

    file_smp_sz.close()

    # Scores to df
    tr_scores_df = scores_to_df( tr_scores_all )
    vl_scores_df = scores_to_df( vl_scores_all )
    te_scores_df = scores_to_df( te_scores_all )

    tr_scores_df.to_csv(gout/'tr_scores.csv', index=False)
    vl_scores_df.to_csv(gout/'vl_scores.csv', index=False)
    te_scores_df.to_csv(gout/'te_scores.csv', index=False)

    if (time()-t0)//3600 > 0:
        lg.logger.info('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        lg.logger.info('Runtime: {:.1f} min'.format( (time()-t0)/60) )
        
    lg.kill_logger()
    # del xdata, ydata


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])


