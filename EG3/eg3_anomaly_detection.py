
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import (
    average_precision_score as ap_score,
    classification_report,
    confusion_matrix,
)
from timeit import default_timer as timer
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from gids_eval.cdr import build_prediction_rows, write_prediction_rows
    from gids_eval.process_monitor import maybe_profile_phase
except Exception:
    build_prediction_rows = None
    write_prediction_rows = None
    maybe_profile_phase = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    p = argparse.ArgumentParser("EG3 PIKACHU")
    p.add_argument("-k", "--dimensions", default=200, type=int)
    p.add_argument("-ip", "--input", default="dataset/cic/cic_5.csv")
    p.add_argument("-l", "--walklen", default=500, type=int)
    p.add_argument("-n", "--numwalk", default=1, type=int)
    p.add_argument("-w", "--trainwin", type=int, default=98)
    p.add_argument("-e", "--epoch", default=50, type=int)
    p.add_argument("-i", "--iter", default=10, type=int)
    p.add_argument("-r", "--alpha", default=0.001, type=float)
    p.add_argument("-s", "--support", default=10, type=int)
    p.add_argument("-t", "--train", nargs="?", const=True, default=True, type=str2bool)
    p.add_argument("-o", "--output", type=str, default="results")
    p.add_argument("-d", "--dataset", type=str, default="lanl")
    p.add_argument("--metrics_csv", type=str, default=None)
    p.add_argument("--gpu", type=str, default=None)

    p.add_argument("--calibration", choices=["transductive", "inductive"], default="transductive")
    p.add_argument("--val_fraction", type=float, default=0.5)
    p.add_argument("--detector_root", type=str, required=True,
                    help="Path to upstream PIKACHU detector directory")
    return p.parse_args()

def aggregate_neighbors(node_emb, node_list, u, n_u):
    CNu = [node_emb[node_list.index(str(n))] for n in n_u]
    Cu = node_emb[node_list.index(str(u))]
    H = 1 / (1 + len(CNu)) * np.add(Cu, sum(Cn for Cn in CNu))
    return np.array(H)

def predict_prob(w, X):
    return softmax(np.dot(X, w.T))

def calculate_edge_probability(w, node_emb, node_list, G):
    edge_scores = []
    for u, v, data in G.edges(data=True):
        Nu = [n for n in G.neighbors(u)]
        Hu = aggregate_neighbors(node_emb, node_list, u, Nu)
        Hu = np.reshape(Hu, (1, Hu.shape[0]))
        Pv = predict_prob(w, Hu)

        Nv = [n for n in G.neighbors(v)]
        Hv = aggregate_neighbors(node_emb, node_list, v, Nv)
        Hv = np.reshape(Hv, (1, Hv.shape[0]))
        Pu = predict_prob(w, Hv)
        score = ((1 - Pv[0, v]) + (1 - Pu[0, u])) / 2
        edge_scores.append([
            u, v, score, data["snapshot"], data["time"],
            data["anom"], data.get("campaign", f"{u}__{v}"),
        ])
    return edge_scores

def select_threshold_transductive(scores, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fw = 0.5
    tw = 1 - fw
    fn = np.abs(tw * tpr - fw * (1 - fpr))
    best = np.argmin(fn, 0)
    return thresholds[best]

def select_threshold_inductive(val_scores, val_labels):
    if len(set(val_labels)) < 2:

        return float(np.percentile(val_scores, 95))
    return select_threshold_transductive(val_scores, val_labels)

def _safe_div(num, den):
    return num / den if den != 0 else 0.0

def compute_and_write_metrics(args, edge_scores_df, threshold, train_time, test_time, out_dir, get_ip_fn):
    true_label = list(edge_scores_df["label"])
    scores = list(edge_scores_df["score"])

    ap = ap_score(true_label, scores) if len(set(true_label)) > 1 else 0.0
    auc_val = metrics.roc_auc_score(true_label, scores) if len(set(true_label)) > 1 else 0.0

    edge_scores_df = edge_scores_df.copy()
    edge_scores_df["pred"] = np.where(edge_scores_df["score"] >= threshold, True, False)
    pred_label = list(edge_scores_df["pred"])
    true_label = list(edge_scores_df["label"])

    tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
    fpr_v = _safe_div(fp, fp + tn)
    tpr_v = _safe_div(tp, tp + fn)
    fnr_v = _safe_div(fn, fn + tp)
    tnr_v = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = tpr_v
    f1 = _safe_div(2 * precision * recall, precision + recall)

    print(f"\n[EG3] calibration={args.calibration}  threshold={threshold:.6f}")
    print(f"      AUC={auc_val:.4f}  AP={ap:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")
    print(f"      TP={tp} FP={fp} FN={fn} TN={tn}")

    os.makedirs(os.path.join(out_dir, "learning"), exist_ok=True)
    o1 = os.path.join(out_dir, "learning", "results.txt")
    with open(o1, "a") as f:
        f.write(f"{args.dataset} (calibration={args.calibration}):\n")
        f.write(f"AUC: {auc_val}\n")
        f.write(f"ap: {ap}\n")
        f.write(f"FPR: {fpr_v}\n")
        f.write(f"TPR: {tpr_v}\n")
        f.write(f"P: {precision}\n")
        f.write(f"tn: {tn}\nfp: {fp}\nfn: {fn}\ntp: {tp}\n")
        f.write(f"cut_off: {threshold}\n")
        f.write(f"tr_time: {train_time/10}\nte_time: {test_time}\n\n")

    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    reported_anom = edge_scores_df[edge_scores_df["pred"] == True].copy()
    reported_anom["src"] = reported_anom["src"].apply(get_ip_fn)
    reported_anom["dest"] = reported_anom["dest"].apply(get_ip_fn)
    result_file = f"{args.dataset}_d{args.dimensions}all_users.csv"
    reported_anom.to_csv(os.path.join(out_dir, "results", result_file))

    metrics_csv = getattr(args, "metrics_csv", None)
    if metrics_csv:
        os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
        row = {
            "dataset": args.dataset,
            "auc": auc_val,
            "ap": ap,
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tpr_v, "fpr": fpr_v, "fnr": fnr_v, "tnr": tnr_v,
            "precision": precision, "recall": recall, "f1": f1,
            "train_time": train_time / 10,
            "test_time": test_time,
            "calibration": args.calibration,
        }
        pd.DataFrame([row]).to_csv(
            metrics_csv, mode="a",
            header=not os.path.exists(metrics_csv),
            index=False,
        )

    predictions_path = os.environ.get("GIDS_PREDICTIONS_PATH")
    if predictions_path and build_prediction_rows and write_prediction_rows:
        prediction_rows = build_prediction_rows(
            dataset=args.dataset,
            scores=edge_scores_df["score"].to_numpy(),
            labels=edge_scores_df["label"].to_numpy(),
            predicted=edge_scores_df["pred"].to_numpy(),
            campaign=edge_scores_df["campaign"].astype(str).to_numpy(),
            eval_index=0,
        )
        write_prediction_rows(predictions_path, prediction_rows, overwrite=True)

if __name__ == "__main__":
    args = parse_args()
    detector_root = Path(args.detector_root)

    sys.path.insert(0, str(detector_root))
    from utils import DataUtils, GraphUtils
    from pikachu import PIKACHU

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    DST = args.output
    file = "sample"
    data_file = "_" + args.dataset + "_" + file + ".pickle"
    os.makedirs(DST, exist_ok=True)
    os.makedirs(os.path.join(DST, "learning"), exist_ok=True)
    os.makedirs(os.path.join(DST, "results"), exist_ok=True)

    weight_file_check = "_" + args.dataset + "_" + file + "_d" + str(args.dimensions) + ".pickle"
    _param_file_check = (
        "weights/param" + "_" + args.dataset + "_" + file
        + "_d" + str(args.dimensions)
        + "_" + str(args.alpha) + "_" + str(args.support) + ".pickle"
    )
    _weights_exist = (
        os.path.exists("weights/node_map" + data_file)
        and os.path.exists("weights/graphs" + data_file)
        and os.path.exists("weights/long_term" + weight_file_check)
        and os.path.exists(_param_file_check)
    )
    if _weights_exist and args.train:
        print("[EG3] Resuming: all weight files found, skipping training phase.")
        args.train = False

    if args.train:
        print("... Parsing Data ... \n")
        dp = DataUtils(data_folder=args.input)
        data_df, node_map = dp.get_data()
        with open("weights/node_map" + data_file, "wb") as f:
            pickle.dump(node_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("... Generating Graphs ... \n")
        g_util = GraphUtils(node_map)
        graphs = []
        for t in tqdm(data_df.snapshot.unique()):
            graphs.append(g_util.create_graph(data_df[data_df["snapshot"] == t]))
        with open("weights/graphs" + data_file, "wb") as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("weights/node_map" + data_file, "rb") as f:
        node_map = pickle.load(f)
    with open("weights/graphs" + data_file, "rb") as f:
        graphs = pickle.load(f)

    print(f"\nTotal Graphs: {len(graphs)}")
    print(len(graphs[0].edges()))
    node_list = [str(v) for k, v in node_map.items()]
    print(f"\nTotal Nodes: {len(node_map)}")

    print("... Starting Graph Embedding ... \n")
    weight_file = "_" + args.dataset + "_" + file + "_d" + str(args.dimensions) + ".pickle"
    if args.train:
        pikachu_model = PIKACHU(args=args, node_list=node_list, node_map=node_map, graphs=graphs)
        pikachu_model.learn_embedding(weight_file)

    print("\n\n =====   EG3 Anomaly Detection ===== ")
    with open("weights/long_term" + weight_file, "rb") as f:
        long_term_embs = pickle.load(f)
    long_term_embs = np.transpose(long_term_embs, (1, 0, 2))

    import random
    from anomaly_detection import AnomalyDetection

    ad = AnomalyDetection(args, node_list, node_map, long_term_embs, idx=0)
    param_file_name = "_" + args.dataset + "_" + file + "_d" + str(args.dimensions)

    learning_rates = [args.alpha]
    support_sets = [args.support]

    def get_ip(node):
        for k, v in node_map.items():
            if v == node:
                return k
        return None

    for s in support_sets:
        train_x, train_y = ad.get_train_edges(graphs[: args.trainwin], args.support)
        for lr in learning_rates:
            ad.args.alpha = lr
            ad.args.support = s
            prob_param_file = param_file_name + "_" + str(lr) + "_" + str(s) + ".pickle"
            prob_param_file = "weights/param" + prob_param_file

            if args.train:
                w = ad.initialize_parameters(args.dimensions, len(node_list))
                ad.idx = 0

                tr_time = timer()
                if maybe_profile_phase:
                    with maybe_profile_phase("train"):
                        parameters, costs = ad.gradient_descent(
                            w, train_x, train_y, args.iter, args.alpha
                        )
                else:
                    parameters, costs = ad.gradient_descent(
                        w, train_x, train_y, args.iter, args.alpha
                    )
                train_time = timer() - tr_time
                w = parameters["w"]
                with open(prob_param_file, "wb") as f:
                    pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                train_time = 0

            with open(prob_param_file, "rb") as f:
                param = pickle.load(f)
            w = param["w"]

            post_train_graphs = graphs[args.trainwin:]
            graph_tuple = [
                (w, long_term_embs[args.trainwin + idx], node_list, G)
                for idx, G in enumerate(post_train_graphs)
            ]

            te_time = timer()
            all_graph_edges = []
            if maybe_profile_phase:
                with maybe_profile_phase("test_0"):
                    for item in tqdm(graph_tuple):
                        all_graph_edges.append(
                            calculate_edge_probability(item[0], item[1], item[2], item[3])
                        )
            else:
                for item in tqdm(graph_tuple):
                    all_graph_edges.append(
                        calculate_edge_probability(item[0], item[1], item[2], item[3])
                    )
            test_time = timer() - te_time

            edge_scores = [e for g_edges in all_graph_edges for e in g_edges]
            edge_scores_df = pd.DataFrame(
                edge_scores,
                columns=["src", "dest", "score", "snapshot", "time", "label", "campaign"],
            )

            if args.calibration == "transductive":

                threshold = select_threshold_transductive(
                    np.array(edge_scores_df["score"]),
                    np.array(edge_scores_df["label"]),
                )
                eval_df = edge_scores_df
            else:

                unique_snapshots = sorted(edge_scores_df["snapshot"].unique())
                n_snapshots = len(unique_snapshots)
                n_val = max(1, int(n_snapshots * args.val_fraction))
                val_snapshots = set(unique_snapshots[:n_val])
                test_snapshots = set(unique_snapshots[n_val:])

                val_df = edge_scores_df[edge_scores_df["snapshot"].isin(val_snapshots)]
                test_df = edge_scores_df[edge_scores_df["snapshot"].isin(test_snapshots)]

                if val_df.empty or test_df.empty:
                    print("[EG3] WARNING: val or test split is empty, falling back to transductive")
                    threshold = select_threshold_transductive(
                        np.array(edge_scores_df["score"]),
                        np.array(edge_scores_df["label"]),
                    )
                    eval_df = edge_scores_df
                else:

                    threshold = select_threshold_inductive(
                        np.array(val_df["score"]),
                        np.array(val_df["label"]),
                    )

                    eval_df = test_df

                print(f"[EG3] inductive split: {n_val} val snapshots, "
                      f"{n_snapshots - n_val} test snapshots")

            compute_and_write_metrics(
                args, eval_df, threshold, train_time, test_time, DST, get_ip,
            )

    del ad
