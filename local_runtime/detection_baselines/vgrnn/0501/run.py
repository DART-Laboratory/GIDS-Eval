import argparse
import os
import pickle
import sys
from copy import deepcopy
from time import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score as auc_score, f1_score, average_precision_score as ap_score
from torch.optim import Adam

from models.serial_models import VGRNN
from loaders.tdata import TData
import loaders.load_oilrig as oilrig
import loaders.load_eg1_0501 as eg1_0501
import loaders.load_sandworm as sandworm
import loaders.load_wizard as wizard
from utils import get_optimal_cutoff, get_score

try:
    from gids_eval.cdr import build_edge_campaigns, build_prediction_rows, write_prediction_rows
    from gids_eval.process_monitor import maybe_profile_phase
except Exception:
    build_edge_campaigns = None
    build_prediction_rows = None
    write_prediction_rows = None
    maybe_profile_phase = None

MIN_EPOCHS = 1
DEFAULT_EPOCHS = 1500
DEFAULT_WORKERS = 8

def _scores_empty(pos_scores, neg_scores):
    return pos_scores.numel() == 0 or neg_scores.numel() == 0

def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0

def _to_float(val):
    if isinstance(val, torch.Tensor):
        return float(val.item())
    return float(val)

def build_outpath():
    root = os.environ.get('VGRNN_RESULTS_ROOT')
    name = os.environ.get('VGRNN_RUN_NAME')
    if root and name:
        return os.path.join(root, name)
    if root:
        return root
    return os.path.join('.', 'results')

def build_model_dir(outpath):
    root = os.environ.get('VGRNN_MODELS_ROOT')
    name = os.environ.get('VGRNN_RUN_NAME')
    if root and name:
        return os.path.join(root, name)
    if root:
        return root
    return os.path.join(outpath, 'models')

def train(model, data, args, model_dir):
    opt = Adam(model.parameters(), lr=float(args.lr))

    best_model = None
    best_score = -1.0
    no_progress = 0
    times = []

    print("Starting training")
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()

        start_t = time()
        forward_args = model.forward(data, TData.TRAIN)
        zs = forward_args[0] if isinstance(model, VGRNN) else forward_args

        pos_edges = [data.ei_masked(TData.TRAIN, i) for i in range(data.T)]
        neg_edges = data.get_negative_edges(TData.TRAIN, nratio=10)
        loss = model.calc_loss(pos_edges, neg_edges, zs)

        loss.backward()
        opt.step()
        elapsed = time() - start_t
        times.append(elapsed)

        print("[%d] Loss: %0.4f\t%0.4fs" % (epoch, loss.item(), elapsed))

        model.eval()
        with torch.no_grad():
            forward_args = model.forward(data, TData.TRAIN)
            zs = forward_args[0] if isinstance(model, VGRNN) else forward_args

            pos_edges = [data.ei_masked(TData.VAL, i) for i in range(data.T)]
            neg_edges = data.get_negative_edges(TData.VAL, nratio=10)
            pos_scores, neg_scores = model.calc_scores(pos_edges, neg_edges, zs)
            if _scores_empty(pos_scores, neg_scores):
                print("\tVal: empty validation edge set, falling back to train split", end="")
                pos_edges = [data.ei_masked(TData.TRAIN, i) for i in range(data.T)]
                neg_edges = data.get_negative_edges(TData.TRAIN, nratio=10)
                pos_scores, neg_scores = model.calc_scores(pos_edges, neg_edges, zs)

            auc, ap = get_score(pos_scores, neg_scores)
            print("\tVal  AUC: %0.4f  AP: %0.4f" % (auc, ap), end='')

            tot = auc + ap
            if tot > best_score:
                no_progress = 0
                best_score = tot
                best_model = deepcopy(model)
                print("*")
            else:
                print()
                if epoch >= MIN_EPOCHS:
                    no_progress += 1

            if no_progress == args.patience:
                print("Early stopping!")
                break

    if best_model is None:
        best_model = model

    model = best_model
    forward_args = model.forward(data, TData.ALL)
    h0 = forward_args[1] if isinstance(model, VGRNN) else None
    tpe = sum(times) / len(times) if times else 0.0
    print("TPE: %0.4f" % tpe)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lr%sd%s.pkl' % (args.lr, args.snapshot))
        with open(model_path, 'wb') as handle:
            pickle.dump({'states': model, 'h0': h0}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, h0, tpe

def find_cutoff(model, data, h0, fw, fallback_data=None):
    if getattr(data, "T", 0) == 0:
        print("No validation time slices found; falling back to train split for cutoff selection")
        if fallback_data is None:
            model.cutoff = 0.5
            return h0
        model.eval()
        with torch.no_grad():
            forward_args = model.forward(fallback_data, TData.TRAIN)
            zs = forward_args[0] if isinstance(model, VGRNN) else forward_args
            pos_edges = [fallback_data.ei_masked(TData.TRAIN, i) for i in range(fallback_data.T)]
            neg_edges = fallback_data.get_negative_edges(TData.TRAIN, nratio=10)
            pos_scores, neg_scores = model.calc_scores(pos_edges, neg_edges, zs)
        model.cutoff = get_optimal_cutoff(pos_scores, neg_scores, fw=fw)
        return h0

    pos_edges = data.eis
    neg_edges = data.get_negative_edges(TData.ALL, nratio=10)

    model.eval()
    with torch.no_grad():
        if isinstance(model, VGRNN):
            zs, h0 = model.forward(data, TData.ALL, h0=h0)
        else:
            zs = model.forward(data, TData.ALL)
            h0 = torch.zeros((1))

        if model.pred:
            pos_edges = pos_edges[1:]
            neg_edges = neg_edges[1:]
            zs = zs[:-1]

        pos_scores, neg_scores = model.calc_scores(pos_edges, neg_edges, zs)
        if _scores_empty(pos_scores, neg_scores):
            print("Validation cutoff edge set empty; falling back to train split for cutoff selection")
            if fallback_data is None:
                model.cutoff = 0.5
                return h0
            forward_args = model.forward(fallback_data, TData.TRAIN)
            train_zs = forward_args[0] if isinstance(model, VGRNN) else forward_args
            pos_edges = [fallback_data.ei_masked(TData.TRAIN, i) for i in range(fallback_data.T)]
            neg_edges = fallback_data.get_negative_edges(TData.TRAIN, nratio=10)
            pos_scores, neg_scores = model.calc_scores(pos_edges, neg_edges, train_zs)
            model.cutoff = get_optimal_cutoff(pos_scores, neg_scores, fw=fw)
            return h0

    model.cutoff = get_optimal_cutoff(pos_scores, neg_scores, fw=fw)
    return h0

def test(model, data, h0, test_time, dataset_label, eval_index):
    pred = 1 if model.pred else 0

    model.eval()
    with torch.no_grad():
        if isinstance(model, VGRNN):
            zs, _ = model.forward(data, TData.ALL, h0=h0)
        else:
            zs = model.forward(data, TData.ALL)

        scores = torch.cat(
            [
                model.decode(
                    data.eis[i + pred][0],
                    data.eis[i + pred][1],
                    zs[i]
                )
                for i in range(data.T - pred)
            ],
            dim=0
        )

    y = torch.cat(data.ys[pred:])
    y_hat = torch.zeros(scores.size(0))
    y_hat[scores <= model.cutoff] = 1

    weights = torch.cat(data.cnt[pred:])
    alerts = y_hat * weights

    tp = _to_float(alerts[y == 1].sum())
    fp = _to_float(alerts[y == 0].sum())
    pos_total = _to_float(weights[y == 1].sum())
    neg_total = _to_float(weights[y == 0].sum())

    fn = pos_total - tp
    tn = neg_total - fp

    precision = _safe_div(tp, tp + fp)
    tpr = _safe_div(tp, pos_total)
    fpr = _safe_div(fp, neg_total)

    y_np = y.cpu().numpy()
    y_hat_np = y_hat.cpu().numpy()
    weights_np = weights.cpu().numpy()
    scores_np = scores.cpu().numpy()
    campaigns = []
    campaign_lists = getattr(data, "campaigns", None)
    if campaign_lists:
        for i in range(pred, data.T):
            campaigns.extend(campaign_lists[i])
    elif build_edge_campaigns:
        for i in range(pred, data.T):
            campaigns.extend(
                build_edge_campaigns(
                    dataset=dataset_label,
                    edge_index=data.eis[i],
                    node_map=getattr(data, "nmap", None),
                )
            )

    f1 = f1_score(y_np, y_hat_np, sample_weight=weights_np)
    auc = auc_score(y_np, 1 - scores_np, sample_weight=weights_np)
    ap = ap_score(y_np, 1 - scores_np, sample_weight=weights_np)

    stats = {
        'Model': 'test',
        'TPR': tpr,
        'FPR': fpr,
        'P': precision,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'F1': f1,
        'AUC': auc,
        'AP': ap,
        'cutoff': float(model.cutoff),
        'FwdTime': test_time,
        'test_time': test_time,
    }
    prediction_rows = []
    if build_prediction_rows and write_prediction_rows:
        prediction_rows = build_prediction_rows(
            dataset=dataset_label,
            scores=1 - scores_np,
            labels=y_np,
            weights=weights_np,
            predicted=y_hat_np,
            campaign=campaigns if campaigns else None,
            eval_index=eval_index,
        )
    return stats, prediction_rows

def run_once(args, loader, tr_start, tr_end, val_start, val_end, te_start, te_end, delta, model_dir, dataset_label, eval_index):
    train_data = loader(args.workers, start=tr_start, end=tr_end, delta=delta)
    model = VGRNN(train_data.xs.size(1), args.dimension, int(args.dimension / 2), pred=False)

    tr_start_t = time()
    if maybe_profile_phase:
        with maybe_profile_phase("train"):
            model, h0, tpe = train(model, train_data, args, model_dir)
    else:
        model, h0, tpe = train(model, train_data, args, model_dir)
    train_time = time() - tr_start_t

    val_data = loader(args.workers, start=val_start, end=val_end, delta=delta)
    h0 = find_cutoff(model, val_data, h0, args.fw, fallback_data=train_data)

    data = loader(args.workers, start=te_start, end=te_end, delta=delta, is_test=True)
    te_start_t = time()
    if maybe_profile_phase:
        with maybe_profile_phase(f"test_{eval_index}"):
            stats, prediction_rows = test(model, data, h0, 0.0, dataset_label, eval_index)
    else:
        stats, prediction_rows = test(model, data, h0, 0.0, dataset_label, eval_index)
    test_time = time() - te_start_t

    stats['TPE'] = tpe
    stats['train_time'] = train_time
    stats['test_time'] = test_time
    stats['FwdTime'] = test_time

    return stats, prediction_rows

def parse_args():
    ap = argparse.ArgumentParser("vgrnn")
    ap.add_argument('-n', '--dataset', type=str, default='oilrig', help='dataset name')
    ap.add_argument('-s', '--snapshot', type=float, default=0.5, help='length of snapshot')
    ap.add_argument('-dim', '--dimension', type=int, default=32, help='dimension')
    ap.add_argument('-l', '--lr', type=float, default=0.005, help='learning rate')
    ap.add_argument('-p', '--patience', type=int, default=10, help='patience')
    ap.add_argument('-f', '--fw', type=float, default=0.5, help='threshold weight')
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='max epochs')
    ap.add_argument('--tests', type=int, default=1, help='number of runs')
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='loader workers')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    delta = int(60 * args.snapshot * 1000)

    ds_name = args.dataset.lower()
    if ds_name == 'oilrig':
        loader = oilrig.load_lanl_dist
        date_of_evil = oilrig.DATE_OF_EVIL_LANL
        te_end = oilrig.TIMES['all']
        ds_label = 'oilrig'
    elif ds_name == 'sandworm':
        loader = sandworm.load_lanl_dist
        date_of_evil = sandworm.DATE_OF_EVIL_LANL
        te_end = sandworm.TIMES['all']
        ds_label = 'sandworm'
    elif ds_name == 'wizard':
        loader = wizard.load_lanl_dist
        date_of_evil = wizard.DATE_OF_EVIL_LANL
        te_end = wizard.TIMES['all']
        ds_label = 'wizard'
    elif ds_name == '0501':
        loader = eg1_0501.load_lanl_dist
        date_of_evil = eg1_0501.DATE_OF_EVIL_LANL
        te_end = eg1_0501.TIMES['all']
        ds_label = '0501'
    else:
        raise NotImplementedError('Only 0501 datasets are supported')

    tr_start = 0
    tr_end = date_of_evil - max((date_of_evil - tr_start) // 20, delta * 2)
    val_start = tr_end
    val_end = date_of_evil
    te_start = date_of_evil

    outpath = build_outpath()
    os.makedirs(outpath, exist_ok=True)
    model_dir = build_model_dir(outpath)

    run_results = [
        run_once(
            args,
            loader,
            tr_start,
            tr_end,
            val_start,
            val_end,
            te_start,
            te_end,
            delta,
            model_dir,
            ds_label,
            eval_index,
        )
        for eval_index in range(args.tests)
    ]
    stats = [item[0] for item in run_results]
    prediction_rows = [row for _, rows in run_results for row in rows]
    predictions_path = os.environ.get("GIDS_PREDICTIONS_PATH")
    if predictions_path and prediction_rows and write_prediction_rows:
        write_prediction_rows(predictions_path, prediction_rows, overwrite=True)

    metrics_df = pd.DataFrame(stats)
    if not metrics_df.empty:
        metrics_df['dataset'] = ds_label
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
        metrics_file = os.path.join(outpath, 'metrics.csv')
        export_df.to_csv(metrics_file, index=False)

    results_file = os.path.join(outpath, 'results.txt')
    with open(results_file, 'w') as handle:
        handle.write(str(args) + '\n')
        handle.write('LR: %s\n' % args.lr)

        if not metrics_df.empty:
            numeric_df = metrics_df.select_dtypes(include=['number'])
            compressed = pd.DataFrame(
                [numeric_df.mean(), numeric_df.sem()],
                index=['mean', 'stderr']
            ).to_csv().replace(',', '\t')
            full = metrics_df.to_csv(index=False, header=False).replace(',', ', ')

            handle.write('test\n')
            handle.write(str(compressed) + '\n')
            handle.write(full + '\n')
