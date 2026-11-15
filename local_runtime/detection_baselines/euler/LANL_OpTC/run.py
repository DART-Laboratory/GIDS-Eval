from argparse import ArgumentParser
import os

import pandas as pd
import torch
import loaders.load_lanl as lanl
import loaders.load_optc as optc
from models.recurrent import GRU, LSTM, EmptyModel
from models.embedders import\
    detector_gcn_rref, detector_gat_rref, detector_sage_rref,\
    predictor_gcn_rref, predictor_gat_rref, predictor_sage_rref

from spinup import run_all
import torch
import numpy as np
import random

DEFAULT_TR = {
    'anom_lr': 0.05,
    'epochs': 100,
    'min': 1,
    'nratio': 10,
    'val_nratio': 1
}

DATA_ROOT = os.environ.get("EULER_DATA_ROOT") or os.path.join(os.environ.get("MIRAGE_INPUT_ROOT", ""), "euler")
RESULTS_ROOT = os.environ.get("EULER_RESULTS_ROOT", os.path.join(DATA_ROOT, 'results'))

def get_args():
    global DEFAULT_TR

    ap = ArgumentParser()

    ap.add_argument(
        '-d', '--delta',
        type=float, default=0.5
    )

    ap.add_argument(
        '-w', '--workers',
        type=int, default=4
    )

    ap.add_argument(
        '-T', '--threads',
        type=int, default=1
    )

    ap.add_argument(
        '-e', '--encoder',
        choices=['GCN', 'GAT', 'SAGE'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM', 'NONE'],
        type=str.upper,
        default="GRU"
    )

    ap.add_argument(
        '-H', '--hidden',
        type=int,
        default=32
    )

    ap.add_argument(
        '-z', '--zdim',
        type=int,
        default=16
    )

    ap.add_argument(
        '-n', '--ngrus',
        type=int,
        default=1
    )

    ap.add_argument(
        '-t', '--tests',
        type=int,
        default=1
    )

    ap.add_argument(
        '-l', '--load',
        action='store_true'
    )

    ap.add_argument(
        '--fpweight',
        type=float,
        default=0.6
    )

    ap.add_argument(
        '--nowrite',
        action='store_true'
    )

    ap.add_argument(
        '--impl', '-i',
        type=str.upper,
        choices=['DETECT', 'PREDICT', 'D', 'P', 'PRED'],
        default="DETECT"
    )

    ap.add_argument(
        '--dataset',
        default='LANL',
        type=str.upper
    )

    ap.add_argument(
        '--lr',
        default=0.01,
        type=float
    )
    ap.add_argument(
        '--patience',
        default=10,
        type=int
    )

    args = ap.parse_args()
    args.te_end = None
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    readable = str(args)
    print(readable)

    model_str = '%s -> %s (%s)' % (args.encoder , args.rnn, args.impl)
    print(model_str)

    if args.dataset.startswith('L'):
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None
        args.te_times = [(args.tr_end, lanl.TIMES['all'])]
        args.delta = int(args.delta * (60**2))
        args.manual = False

    elif args.dataset.startswith('O'):
        args.loader = optc.load_lanl_dist
        args.tr_start = 0
        args.tr_end = optc.DATE_OF_EVIL_LANL
        args.val_times = None
        args.te_times = [(args.tr_end, optc.TIMES['all'])]
        args.delta = int(args.delta * (60**2))
        args.manual = False

        _tr_start = os.environ.get('EULER_TR_START')
        _tr_end = os.environ.get('EULER_TR_END')
        _te_end = os.environ.get('EULER_TE_END')
        if _tr_start is not None:
            args.tr_start = int(_tr_start)
        if _tr_end is not None:
            args.tr_end = int(_tr_end)
        if _tr_end is not None or _te_end is not None:
            te_end_val = int(_te_end) if _te_end is not None else optc.TIMES['all']
            args.te_times = [(args.tr_end, te_end_val)]

    else:
        raise NotImplementedError('Only the LANL data set is supported in this release')

    if args.encoder == 'GCN':
        args.encoder = detector_gcn_rref if args.impl[0] == 'D'\
            else predictor_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = detector_gat_rref if args.impl[0] == 'D'\
            else predictor_gat_rref
    else:
        args.encoder = detector_sage_rref if args.impl[0] == 'D'\
            else predictor_sage_rref

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM
    else:
        args.rnn = EmptyModel

    return args, readable, model_str

if __name__ == '__main__':
    args, argstr, modelstr = get_args()
    DEFAULT_TR['lr'] = args.lr
    DEFAULT_TR['patience'] = args.patience

    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]

    ds_name = args.dataset.lower()
    out_dir = os.path.join(RESULTS_ROOT, ds_name, 'learning')
    os.makedirs(out_dir, exist_ok=True)
    import spinup
    spinup.MODEL_PATH = os.path.join(out_dir, 'model_save.pkl')
    spinup.TMP_FILE = os.path.join(out_dir, 'tmp.dat')
    spinup.SCORE_FILE = os.path.join(out_dir, 'scores.txt')

    stats = [
        run_all(
            args.workers,
            args.rnn,
            rnn_args,
            args.encoder,
            worker_args,
            args.delta,
            args.load,
            args.fpweight,
            args.impl,
            args.loader,
            args.tr_start,
            args.tr_end,
            args.val_times,
            args.te_times,
            DEFAULT_TR
        )
        for _ in range(args.tests)
    ]

    if args.nowrite:
        exit()

    flat_stats = [item for run_stats in stats for item in run_stats]
    metrics_df = pd.DataFrame(flat_stats)
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
        metrics_file = os.path.join(out_dir, 'metrics.csv')
        export_df.to_csv(metrics_file, index=False)

    results_file = os.path.join(out_dir, 'results.txt')
    f = open(results_file, 'a')
    f.write(str(argstr) + '\n')
    f.write('LR: ' + str(args.lr) + '\n')
    f.write(modelstr + '\n')

    dfs = [pd.DataFrame(s) for s in list(zip(*stats))]
    dfs = pd.concat(dfs, axis=0)

    for m in dfs['Model'].unique():
        df = dfs[dfs['Model'] == m]

        numeric_df = df.select_dtypes(include=['number'])
        compressed = pd.DataFrame(
            [numeric_df.mean(), numeric_df.sem()],
            index=['mean', 'stderr']
        ).to_csv().replace(',', '\t')

        full = df.to_csv(index=False, header=False)
        full = full.replace(',', ', ')

        f.write(m + '\n')
        f.write(str(compressed) + '\n')
        f.write(full + '\n')

    f.close()
