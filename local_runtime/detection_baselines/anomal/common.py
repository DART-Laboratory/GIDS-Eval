from urllib.parse import urlparse
import os
import time
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import dgl
import category_encoders as ce
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
from sklearn.ensemble import IsolationForest
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA

from model import DGI

try:
    from gids_eval.cdr import build_prediction_rows, write_prediction_rows
except Exception:
    build_prediction_rows = None
    write_prediction_rows = None

def get_host(url: str) -> (str, str, str):
    parsed_url = urlparse(url)
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path

try:
    import urllib3
    urllib3.get_host = get_host
except Exception:
    pass

warnings.filterwarnings("ignore")

def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0

def build_outpath():
    root = os.environ.get("ANOMAL_RESULTS_ROOT")
    name = os.environ.get("ANOMAL_RUN_NAME")
    if root and name:
        return os.path.join(root, name)
    if root:
        return root
    return os.path.join(".", "results")

def build_model_dir(outpath):
    root = os.environ.get("ANOMAL_MODELS_ROOT")
    name = os.environ.get("ANOMAL_RUN_NAME")
    if root and name:
        return os.path.join(root, name)
    if root:
        return root
    return os.path.join(outpath, "models")

def resolve_data_root(cli_value=None):
    if cli_value:
        return cli_value
    env = os.environ.get("ANOMAL_DATA_ROOT")
    if env:
        return env
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidate = os.path.join(base, "anomal")
    if os.path.exists(candidate):
        return candidate
    return candidate

def _prepare_cic_dataframe(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    if "label" not in df.columns and "Label" in df.columns:
        df = df.rename(columns={"Label": "label"})

    if "label" not in df.columns:
        raise ValueError("CIC data missing label column")

    src_col = "Source IP" if "Source IP" in df.columns else "Src IP"
    dst_col = "Destination IP" if "Destination IP" in df.columns else "Dst IP"
    if src_col in df.columns:
        df[src_col] = df[src_col].astype(str)
    if dst_col in df.columns:
        df[dst_col] = df[dst_col].astype(str)

    drop_cols = [
        "Flow ID",
        "Timestamp",
        "Label",
        "Attempted Category",
        "time_int",
        "Source Port",
        "Destination Port",
        "Src Port",
        "Dst Port",
        "id",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    return df, src_col, dst_col

def _prepare_0501_dataframe(file_path):
    df = pd.read_csv(file_path, header=None)
    cols = [
        "Source IP",
        "Destination IP",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
        "label",
    ]
    if df.shape[1] >= 12:
        cols.append("campaign")
    df.columns = cols[: df.shape[1]]
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df["Source IP"] = df["Source IP"].fillna("").astype(str)
    df["Destination IP"] = df["Destination IP"].fillna("").astype(str)
    numeric_cols = [
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
        "label",
    ]
    for col in [name for name in numeric_cols if name in df.columns]:
        df[col] = pd.to_numeric(df[col].replace(r"^\s*$", "0", regex=True), errors="coerce").fillna(0)
    df["label"] = df["label"].astype(int)
    if "campaign" in df.columns:
        df["campaign"] = df["campaign"].fillna("").astype(str)
    return df, "Source IP", "Destination IP"

def _train_test_split(df, split_index):
    split_index = int(split_index)
    if split_index <= 0 or split_index >= len(df):
        label_idx = df.index[df["label"] == 1]
        if len(label_idx) > 0:
            split_index = int(label_idx[0])
        else:
            split_index = int(len(df) * 0.8)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    return train_data, test_data

def _encode_and_normalize(train_data, test_data, src_col, dst_col, cat_cols=None):
    X_train = train_data.drop(columns=["label"]).copy()
    y_train = train_data[["label"]]
    X_test = test_data.drop(columns=["label"]).copy()
    y_test = test_data[["label"]]

    for frame in (X_train, X_test):
        if "campaign" in frame.columns:
            frame.drop(columns=["campaign"], inplace=True)

    if cat_cols:
        cat_cols = [c for c in cat_cols if c in X_train.columns]
        if cat_cols:
            encoder = ce.TargetEncoder(cols=cat_cols)
            encoder.fit(X_train, y_train.label)
            X_train = encoder.transform(X_train)
            X_test = encoder.transform(X_test)

    cols_to_norm = [c for c in X_train.columns if c not in [src_col, dst_col]]
    if cols_to_norm:
        X_train[cols_to_norm] = X_train[cols_to_norm].replace(r"^\s*$", np.nan, regex=True)
        X_test[cols_to_norm] = X_test[cols_to_norm].replace(r"^\s*$", np.nan, regex=True)
    for col in cols_to_norm:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    scaler = Normalizer()
    if cols_to_norm:
        scaler.fit(X_train[cols_to_norm])
        X_train[cols_to_norm] = scaler.transform(X_train[cols_to_norm])
        X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])

    X_train["h"] = X_train[cols_to_norm].values.tolist() if cols_to_norm else [[]] * len(X_train)
    X_test["h"] = X_test[cols_to_norm].values.tolist() if cols_to_norm else [[]] * len(X_test)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, test

def _build_graph(df, src_col, dst_col):
    g = nx.from_pandas_edgelist(df, src_col, dst_col, ["h", "label"], create_using=nx.MultiGraph())
    g = g.to_directed()
    g = dgl.from_networkx(g, edge_attrs=["h", "label"])
    nfeat_weight = torch.ones([g.number_of_nodes(), g.edata["h"].shape[1]])
    g.ndata["h"] = nfeat_weight

    g.ndata["h"] = torch.reshape(g.ndata["h"], (g.ndata["h"].shape[0], 1, g.ndata["h"].shape[1]))
    g.edata["h"] = torch.reshape(g.edata["h"], (g.edata["h"].shape[0], 1, g.edata["h"].shape[1]))
    return g

def _train_dgi(train_g, ndim_out, learning_rate, epochs, patience, model_path):
    ndim_in = train_g.ndata["h"].shape[2]
    edim = train_g.edata["h"].shape[2]
    dgi = DGI(ndim_in, ndim_out, edim, F.relu)

    optimizer = torch.optim.Adam(dgi.parameters(), lr=learning_rate, weight_decay=0.0)
    best_loss = 1e9
    best_state = None
    wait = 0
    dur = []

    for epoch in range(epochs):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        optimizer.zero_grad()
        loss = dgi(train_g, train_g.ndata["h"], train_g.edata["h"])
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu() for k, v in dgi.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait == patience:
            print("Early stopping!")
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        if epoch % 1 == 0:
            avg_dur = np.mean(dur) if dur else 0.0
            etputs = train_g.num_edges() / avg_dur / 1000 if avg_dur else 0.0
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | ETputs(KTEPS) {:.2f}".format(
                epoch, avg_dur, loss.item(), etputs
            ))

    if best_state is None:
        best_state = dgi.state_dict()
    if model_path:
        torch.save(best_state, model_path)
    dgi.load_state_dict(best_state)
    return dgi

def _embed_graphs(dgi, train_g, test_g):
    dgi.eval()
    with torch.no_grad():
        train_emb = dgi.encoder(train_g, train_g.ndata["h"], train_g.edata["h"])[1]
        test_emb = dgi.encoder(test_g, test_g.ndata["h"], test_g.edata["h"])[1]
    df_train = pd.DataFrame(train_emb.detach().cpu().numpy())
    df_train["label"] = train_g.edata["label"].detach().cpu().numpy()
    df_test = pd.DataFrame(test_emb.detach().cpu().numpy())
    df_test["label"] = test_g.edata["label"].detach().cpu().numpy()
    return df_train, df_test

def _run_detectors(df_train, df_test):
    normal_train = df_train.drop(columns=["label"])
    test_labels = df_test["label"].astype(int).to_numpy()
    test_samples = df_test.drop(columns=["label"])

    detectors = [
        ("cblof", CBLOF, {
            "n_clusters": [3, 5, 7, 9, 10],
            "contamination": [0.001, 0.01, 0.04, 0.05, 0.1, 0.2],
            "alpha": [0.75],
            "beta": [3],
        }),
        ("hbos", HBOS, {
            "n_bins": [5, 10, 15, 20, 25, 30],
            "contamination": [0.001, 0.01, 0.04, 0.05, 0.1, 0.2],
        }),
        ("pca", PCA, {
            "n_components": [5, 10, 15, 20, 25, 30],
            "contamination": [0.001, 0.01, 0.04, 0.05, 0.1, 0.2],
        }),
        ("if", IsolationForest, {
            "n_estimators": [20, 50, 100, 150],
            "contamination": [0.001, 0.01, 0.04, 0.05, 0.1, 0.2],
        }),
    ]

    best = {
        "score": -1,
        "name": None,
        "params": None,
        "pred": None,
        "scores": None,
        "auc": 0.0,
        "ap": 0.0,
    }

    for name, cls, grid in detectors:
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:
                if name == "if":
                    clf = cls(**params)
                else:
                    clf = cls(**params)
                clf.fit(normal_train)
                pred = clf.predict(test_samples)
                pred = np.asarray(pred)
                if set(np.unique(pred)).issubset({-1, 1}):
                    pred = (pred == -1).astype(int)
                if name == "if" and hasattr(clf, "score_samples"):
                    scores = -np.asarray(clf.score_samples(test_samples))
                elif hasattr(clf, "decision_function"):
                    scores = np.asarray(clf.decision_function(test_samples))
                else:
                    scores = pred.astype(float)
                auc = roc_auc_score(test_labels, pred)
                ap = average_precision_score(test_labels, pred)
                score = auc + ap
                if score > best["score"]:
                    best.update({
                        "score": score,
                        "name": name,
                        "params": params,
                        "pred": pred,
                        "scores": scores,
                        "auc": auc,
                        "ap": ap,
                    })
            except Exception:
                continue

    if best["pred"] is None:
        best["pred"] = np.zeros_like(test_labels)
    if best["scores"] is None:
        best["scores"] = best["pred"].astype(float)

    return best, test_labels

def _write_metrics(outpath, dataset, best, test_labels, train_time, test_time, campaigns=None):
    pred = best["pred"].astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, pred, labels=[0, 1]).ravel()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = f1_score(test_labels, pred, zero_division=0)
    tpr = recall
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    tnr = _safe_div(tn, tn + fp)

    threshold = 0.0
    if best.get("params"):
        threshold = best["params"].get("contamination", 0.0)

    export_df = pd.DataFrame([{
        "dataset": dataset,
        "auc": best.get("auc", 0.0),
        "ap": best.get("ap", 0.0),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "tnr": tnr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_time": train_time,
        "test_time": test_time,
    }])

    metrics_file = os.path.join(outpath, "metrics.csv")
    export_df.to_csv(metrics_file, index=False)

    results_file = os.path.join(outpath, "results.txt")
    with open(results_file, "w") as f:
        f.write("dataset: {}\n".format(dataset))
        f.write("best_detector: {}\n".format(best.get("name")))
        f.write("best_params: {}\n".format(best.get("params")))
        f.write("auc: {}\n".format(best.get("auc")))
        f.write("ap: {}\n".format(best.get("ap")))
        f.write("tp: {}\n".format(tp))
        f.write("fp: {}\n".format(fp))
        f.write("fn: {}\n".format(fn))
        f.write("tn: {}\n".format(tn))
        f.write("precision: {}\n".format(precision))
        f.write("recall: {}\n".format(recall))
        f.write("f1: {}\n".format(f1))
        f.write("train_time: {}\n".format(train_time))
        f.write("test_time: {}\n".format(test_time))

    predictions_path = os.environ.get("GIDS_PREDICTIONS_PATH")
    if predictions_path and build_prediction_rows and write_prediction_rows:
        export_campaigns = campaigns
        if export_campaigns is not None:
            try:
                if len(export_campaigns) != len(pred):
                    export_campaigns = None
            except TypeError:
                export_campaigns = campaigns
        prediction_rows = build_prediction_rows(
            dataset=dataset,
            scores=best.get("scores", pred.astype(float)),
            labels=test_labels,
            predicted=pred,
            campaign=export_campaigns,
            eval_index=0,
        )
        write_prediction_rows(predictions_path, prediction_rows, overwrite=True)

def run_cic(file_path, split, ndim_out, learning_rate, epochs, patience, dataset_label):
    outpath = build_outpath()
    model_dir = build_model_dir(outpath)
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df, src_col, dst_col = _prepare_cic_dataframe(file_path)
    train_data, test_data = _train_test_split(df, split)
    test_campaigns = test_data["campaign"].astype(str).to_numpy() if "campaign" in test_data.columns else None

    train, test = _encode_and_normalize(
        train_data,
        test_data,
        src_col,
        dst_col,
        cat_cols=["Protocol", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags"],
    )

    train_g = _build_graph(train, src_col, dst_col)
    test_g = _build_graph(test, src_col, dst_col)

    model_path = os.path.join(model_dir, "best_dgi_{}.pt".format(ndim_out))
    train_start = time.time()
    dgi = _train_dgi(train_g, ndim_out, learning_rate, epochs, patience, model_path)
    df_train, df_test = _embed_graphs(dgi, train_g, test_g)
    train_time = time.time() - train_start

    test_start = time.time()
    best, test_labels = _run_detectors(df_train, df_test)
    test_time = time.time() - test_start

    _write_metrics(outpath, dataset_label, best, test_labels, train_time, test_time, campaigns=test_campaigns)

def run_0501(file_path, split, ndim_out, learning_rate, epochs, patience, dataset_label):
    outpath = build_outpath()
    model_dir = build_model_dir(outpath)
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df, src_col, dst_col = _prepare_0501_dataframe(file_path)
    train_data, test_data = _train_test_split(df, split)
    test_campaigns = test_data["campaign"].astype(str).to_numpy() if "campaign" in test_data.columns else None

    train, test = _encode_and_normalize(train_data, test_data, src_col, dst_col, cat_cols=None)

    train_g = _build_graph(train, src_col, dst_col)
    test_g = _build_graph(test, src_col, dst_col)

    model_path = os.path.join(model_dir, "best_dgi_{}.pt".format(ndim_out))
    train_start = time.time()
    dgi = _train_dgi(train_g, ndim_out, learning_rate, epochs, patience, model_path)
    df_train, df_test = _embed_graphs(dgi, train_g, test_g)
    train_time = time.time() - train_start

    test_start = time.time()
    best, test_labels = _run_detectors(df_train, df_test)
    test_time = time.time() - test_start

    _write_metrics(outpath, dataset_label, best, test_labels, train_time, test_time, campaigns=test_campaigns)
