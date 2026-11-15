from argparse import ArgumentParser
import os, datetime
import pandas as pd
import torch
import loaders.load_optc as optc
import loaders.load_lanl as lanl
from models.recurrent import GRU, LSTM, EmptyModel
from models.argus import detector_lanl_rref, detector_optc_rref
from classification import classification

import numpy as np

def args():
    ap = ArgumentParser()
    ap.add_argument('-d', '--delta', type=float, default=1)
    ap.add_argument('-e', '--encoder_name', type=str.upper,default="ARGUS")
    ap.add_argument('-r', '--rnn', choices=['GRU', 'LSTM', 'NONE'], type=str.upper, default="GRU")
    ap.add_argument('-H', '--hidden', type=int, default=32)
    ap.add_argument('-z', '--zdim', type=int, default=16)
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('--gpu', action='store_true')

    ap.add_argument('-te', '--te_end', choices=['20', '100', '500', 'all', 'test'], type=str.lower, default="test")
    ap.add_argument('--fpweight', type=float, default=0.6)

    ap.add_argument('--dataset', default='LANL', type=str.upper, choices=['OPTC', 'LANL'])
    ap.add_argument('--lr', default=0.01, type=float)
    ap.add_argument('--patience', default=3, type=int)
    ap.add_argument('--nratio', default=1, type=int)
    ap.add_argument('--epochs', default=100, type=int)
    ap.add_argument('--flows', action='store_false')
    args = ap.parse_args()
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'
    readable = str(args)
    print(readable)
    model_str = '%s -> %s ' % (args.encoder_name , args.rnn)
    print(model_str)

    if args.dataset.startswith('O'):
        args.loader = optc.load_optc_dist
        args.tr_start = 0
        args.tr_end = optc.DATE_OF_EVIL_LANL
        args.val_times = None

        args.te_times = [(args.tr_end, optc.TIMES[args.te_end])]
        args.delta = int(args.delta * (60**2))
    elif args.dataset.startswith('L'):
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None

        args.te_times = [(args.tr_end, lanl.TIMES[args.te_end])]

        args.delta = int(args.delta * (60**2))

    else:
        raise NotImplementedError('Only OpTC and LANL data sets are supported.')

    if (args.encoder_name == 'ARGUS') and (args.dataset.startswith('L')):
        args.encoder = detector_lanl_rref
    elif (args.encoder_name == 'ARGUS') and (args.dataset.startswith('O')):
        args.encoder = detector_optc_rref
    else:
        raise NotImplementedError("wrong encoder", args.encoder_name, args.dataset)

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM
    else:
        args.rnn = EmptyModel
    return args, readable, model_str

def build_outpath():
    root = os.environ.get('ARGUS_RESULTS_ROOT')
    name = os.environ.get('ARGUS_RUN_NAME')
    if root and name:
        return os.path.join(root, name)
    if root:
        return root
    return os.path.join('.', 'Exps', 'result')

if __name__ == '__main__':
    args, argstr, modelstr = args()
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(device)
    OUTPATH = build_outpath()
    if not os.path.exists(OUTPATH): os.makedirs(OUTPATH)
    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]
    stats = classification(args, rnn_args, worker_args, OUTPATH, device)
    ds_name = args.dataset.lower()
    if ds_name.startswith('o_') or ds_name.startswith('l_'):
        ds_name = ds_name[2:]
    metrics_df = pd.DataFrame(stats)
    if not metrics_df.empty:
        metrics_df['dataset'] = ds_name
        if 'train_time' not in metrics_df:
            metrics_df['train_time'] = metrics_df.get('tr_time', 0)
        if 'test_time' not in metrics_df:
            metrics_df['test_time'] = metrics_df.get('FwdTime', 0)
        if 'precision' not in metrics_df:
            metrics_df['precision'] = metrics_df.get('P', 0)
        if 'recall' not in metrics_df:
            metrics_df['recall'] = metrics_df.get('TPR', 0)
        if 'threshold' not in metrics_df:
            metrics_df['threshold'] = metrics_df.get('cutoff', 0)
        if 'TP' not in metrics_df:
            metrics_df['TP'] = metrics_df.get('tp', 0)
        if 'FP' not in metrics_df:
            metrics_df['FP'] = metrics_df.get('fp', 0)
        if 'FN' not in metrics_df:
            metrics_df['FN'] = metrics_df.get('fn', 0)
        if 'TN' not in metrics_df:
            metrics_df['TN'] = metrics_df.get('tn', 0)
        if 'FNR' not in metrics_df:
            denom = metrics_df['FN'] + metrics_df['TP']
            metrics_df['FNR'] = metrics_df['FN'] / denom.replace(0, np.nan)
            metrics_df['FNR'] = metrics_df['FNR'].fillna(0)
        if 'TNR' not in metrics_df:
            denom = metrics_df['TN'] + metrics_df['FP']
            metrics_df['TNR'] = metrics_df['TN'] / denom.replace(0, np.nan)
            metrics_df['TNR'] = metrics_df['TNR'].fillna(0)
        export_df = pd.DataFrame({
            'dataset': metrics_df['dataset'],
            'auc': metrics_df['AUC'],
            'ap': metrics_df['AP'],
            'threshold': metrics_df['threshold'],
            'tp': metrics_df['TP'],
            'fp': metrics_df['FP'],
            'fn': metrics_df['FN'],
            'tn': metrics_df['TN'],
            'tpr': metrics_df['TPR'],
            'fpr': metrics_df['FPR'],
            'fnr': metrics_df['FNR'],
            'tnr': metrics_df['TNR'],
            'precision': metrics_df['precision'],
            'recall': metrics_df['recall'],
            'f1': metrics_df['F1'],
            'train_time': metrics_df['train_time'],
            'test_time': metrics_df['test_time'],
        })
        metrics_file = os.path.join(OUTPATH, 'metrics.csv')
        export_df.to_csv(metrics_file, index=False)
    f = open(os.path.join(OUTPATH, 'results.txt'), 'w')
    f.write('\n')
    f.write(str(argstr) + '\n')
    f.write('LR: ' + str(args.lr) + '\n')
    f.write(str(stats[0])+'\n')
