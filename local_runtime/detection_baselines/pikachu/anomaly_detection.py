from tqdm import tqdm

from sklearn.metrics import confusion_matrix, log_loss, classification_report
from sklearn.metrics import average_precision_score as ap_score
from sklearn import metrics
import numpy as np
import pandas as pd
import random
from scipy.special import softmax
import pickle
from multiprocessing import Pool

from timeit import default_timer as timer
import os
import tensorflow as tf

try:
    from gids_eval.cdr import build_prediction_rows, write_prediction_rows
    from gids_eval.process_monitor import maybe_profile_phase
except Exception:
    build_prediction_rows = None
    write_prediction_rows = None
    maybe_profile_phase = None

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
        if 1==1:
            Nu = [n for n in G.neighbors(u)]
            Hu = aggregate_neighbors(node_emb, node_list, u, Nu)
            Hu = np.reshape(Hu, (1, Hu.shape[0]))
            Pv = predict_prob(w, Hu)

            Nv = [n for n in G.neighbors(v)]
            Hv = aggregate_neighbors(node_emb, node_list, v, Nv)
            Hv = np.reshape(Hv, (1, Hv.shape[0]))
            Pu = predict_prob(w, Hv)
            score = ((1-Pv[0, v]) + (1-Pu[0, u]))/2
            edge_scores.append([u, v, score, data['snapshot'], data['time'], data['anom'], data.get('campaign', f"{u}__{v}")])
    return edge_scores

class AnomalyDetection:
    def __init__(self, args, node_list, node_map, node_embeddings, idx):
        self.args = args
        self.node_list = node_list
        self.node_map = node_map
        self.node_embeddings = node_embeddings
        self.idx = idx

    def get_ip(self, node):
        for k, v in self.node_map.items():
            if v == node:
                return k
        return None

    def aggregate_neighbors_object(self, u, n_u):
        CNu = [self.node_embeddings[self.idx][self.node_list.index(str(n))] for n in n_u]
        Cu = self.node_embeddings[self.idx][self.node_list.index(str(u))]
        H = 1/(1+len(CNu))*np.add(Cu, sum(Cn for Cn in CNu))
        return np.array(H)

    def initialize_parameters(self, k, v):
        w = np.random.randn(v, k) * 0.0001
        return w

    def propagate(self, w, X, Y):
        m = X.shape[1]
        p = softmax(np.dot(X, w.T))
        cost = log_loss(Y, p)

        dw = (1 / m) * np.dot((p - Y).T, X)

        cost = np.squeeze(cost)
        grads = {"dw": dw}
        return grads, cost

    def gradient_descent(self, w, X, Y, iterations, learning_rate):
        costs = []
        print("start costs")
        for i in range(iterations):
            grads, cost = self.propagate(w, X, Y)
            w = w - learning_rate * grads["dw"]
            costs.append(cost)
            print("Cost after iteration %i/%i:      %f" % (i, iterations, cost))
        params = {"w": w}
        return params, costs

    def predict(self, w, X):
        return softmax(np.dot(X, w.T))

    def get_train_edges(self, train_graphs, s = 10):
        data_x = []
        data_y = []
        for G in tqdm(train_graphs):
            print("G: ", len(G.nodes()), len(G.edges()))
            for u in G.nodes():
                N = [n for n in G.neighbors(u)]
                for v in N:
                    if len(N) > 1:
                        n_minus_v = [n for n in N if n != v]
                        support_set = random.choices(n_minus_v, k=s)
                    else:
                        support_set = random.choices(N, k=s)
                    H = self.aggregate_neighbors_object(u, support_set)
                    y = np.zeros(len(self.node_list))
                    y[v] = 1
                    data_x.append(H)
                    data_y.append(y)
            self.idx += 1

        return np.array(data_x), np.array(data_y)

    def print_result(self, percentile, threshold, true_label, pred_label):
        print("\n====BEST ANOMALY DETECTION RESULTS====")
        print("Percentile : ", percentile, " Threshold : ", threshold)
        print("---------------------------------------\n")
        print(metrics.classification_report(true_label, pred_label))
        print("Confusion Matrix: \n", confusion_matrix(true_label, pred_label, labels=[False, True]))
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
        print("(tn, fp, fn, tp): ", tn, fp, fn, tp)

    def _safe_div(self, num, den):
        return num / den if den != 0 else 0.0

    def calculate_performance_metrics(self, edge_scores, result_file,tr,te, out_dir):
        print("\n\nCalculating Performance Metrices....")
        true_label = list(edge_scores['label'])
        scores = list(edge_scores['score'])
        fpr, tpr, thresholds = metrics.roc_curve(true_label, scores, pos_label=1)
        ap = ap_score(true_label, scores)
        fw = 0.5
        tw = 1 - fw
        fn = np.abs(tw * tpr - fw * (1 - fpr))
        best = np.argmin(fn, 0)
        print("\n\nOptimal cutoff %0.10f achieves TPR: %0.5f FPR: %0.5f on train data"
              % (thresholds[best], tpr[best], fpr[best]))
        print("Final AUC: ", metrics.auc(fpr, tpr))
        print("AUC: ", metrics.roc_auc_score(true_label, scores))
        print("AP:" ,ap)
        edge_scores['pred'] = np.where(edge_scores['score'] >= thresholds[best], True, False)
        true_label = list(edge_scores['label'])
        pred_label = list(edge_scores['pred'])
        print("\n\n======= CLASSIFICATION REPORT =========\n")
        print(classification_report(true_label, pred_label))
        tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
        print("Confusion Matrix: \n", confusion_matrix(true_label, pred_label, labels=[False, True]))
        fpr_v = self._safe_div(fp, (fp + tn))
        tpr_v = self._safe_div(tp, (tp + fn))
        fnr_v = self._safe_div(fn, (fn + tp))
        tnr_v = self._safe_div(tn, (tn + fp))
        precision = self._safe_div(tp, (tp + fp))
        recall = tpr_v
        f1 = self._safe_div(2 * precision * recall, (precision + recall))

        print("FPR: ", fpr_v)
        print("TPR: ", tpr_v)
        print("P: %0.4f, R: %0.4f" % (precision, recall))
        o1 = os.path.join(out_dir, "learning", "results.txt")
        f= open(o1,'a')
        f.write(self.args.dataset+':'+'\n')
        f.write("AUC: "+str(metrics.roc_auc_score(true_label, scores))+'\n')
        f.write("ap: "+str(ap)+'\n')
        f.write("FPR: "+str(fpr_v)+'\n')
        f.write("TPR: "+str(tpr_v)+'\n')
        f.write("P: "+str(precision)+'\n')
        f.write("tn: "+str(tn)+'\n')
        f.write("fp: "+str(fp)+'\n')
        f.write("fn: "+str(fn)+'\n')
        f.write("tp: "+str(tp)+'\n')
        f.write("cut_off: "+str(thresholds[best])+'\n')
        f.write("tr_time: "+str(tr/10)+'\n')
        f.write("te_time: "+str(te)+'\n')
        f.write('\n')
        f.close()

        reported_anom = edge_scores[edge_scores['pred'] == True]
        reported_anom['src'] = reported_anom['src'].apply(self.get_ip)
        reported_anom['dest'] = reported_anom['dest'].apply(self.get_ip)
        o2 = os.path.join(out_dir,"results", result_file)
        reported_anom.to_csv(o2)

        metrics_csv = getattr(self.args, "metrics_csv", None)
        if metrics_csv:
            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            row = {
                "dataset": self.args.dataset,
                "auc": metrics.roc_auc_score(true_label, scores),
                "ap": ap,
                "threshold": thresholds[best],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tpr": tpr_v,
                "fpr": fpr_v,
                "fnr": fnr_v,
                "tnr": tnr_v,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_time": tr/10,
                "test_time": te,
            }
            pd.DataFrame([row]).to_csv(metrics_csv, mode='a', header=not os.path.exists(metrics_csv), index=False)

        predictions_path = os.environ.get("GIDS_PREDICTIONS_PATH")
        if predictions_path and build_prediction_rows and write_prediction_rows:
            prediction_rows = build_prediction_rows(
                dataset=self.args.dataset,
                scores=edge_scores["score"].to_numpy(),
                labels=edge_scores["label"].to_numpy(),
                predicted=edge_scores["pred"].to_numpy(),
                campaign=edge_scores["campaign"].astype(str).to_numpy(),
                eval_index=0,
            )
            write_prediction_rows(predictions_path, prediction_rows, overwrite=True)

    def anomaly_detection(self, graphs, param_file, out_dir):
        print("\n\nEstimating Edge Probability Distribution....")
        learning_rates = [self.args.alpha]
        support_sets = [self.args.support]
        i=0
        for s in support_sets:
            train_x, train_y = self.get_train_edges(graphs[:self.args.trainwin], self.args.support)
            for lr in learning_rates:
                learn_out_file = os.path.join(out_dir, "learning", "results.txt")
                f= open(learn_out_file,'a')
                f.write("lr: "+str(lr)+'\n')
                f.write("s: "+str(s)+'\n')
                f.close()
                self.args.alpha = lr
                self.args.support = s
                prob_param_file = param_file + '_' + str(self.args.alpha) + '_' + str(self.args.support) + '.pickle'
                print("++++++++++ Parameters +++++++ ")
                print("Learning Rate: ", lr)
                print("# of Support Set: ", s)
                print("Param File: ", prob_param_file)
                if self.args.train:
                    w = self.initialize_parameters(self.args.dimensions, len(self.node_list))

                    print("\n\nGenerating Training Edges....")
                    self.idx = 0

                    print("\n\nStarting Gradient Descent....")
                    tr_time = timer()
                    print("started")
                    if maybe_profile_phase:
                        with maybe_profile_phase("train"):
                            parameters, costs = self.gradient_descent(w, train_x, train_y, self.args.iter, self.args.alpha)
                    else:
                        parameters, costs = self.gradient_descent(w, train_x, train_y, self.args.iter, self.args.alpha)
                    print("finished")
                    train_time=timer() - tr_time
                    print(train_time)
                    w = parameters["w"]

                    with open(prob_param_file, 'wb') as f:
                        pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

                with open(prob_param_file, 'rb') as f:
                    param = pickle.load(f)
                w = param['w']
                total_cpu = os.cpu_count()
                print("\nNumber of CPU Available: ", total_cpu)

                graph_tuple = [(w, self.node_embeddings[self.args.trainwin + idx], self.node_list, G) for idx, G in enumerate(graphs[self.args.trainwin:])]
                te_time = timer()
                train_time=0

                all_graph_edges=[]
                if maybe_profile_phase:
                    with maybe_profile_phase("test_0"):
                        for i in tqdm(graph_tuple):
                            all_graph_edges.append(calculate_edge_probability(i[0],i[1],i[2],i[3]))
                else:
                    for i in tqdm(graph_tuple):
                        all_graph_edges.append(calculate_edge_probability(i[0],i[1],i[2],i[3]))

                print("\nEdge Probability Estimation Completed...   [%s Sec.]" % (timer() - te_time))
                test_time=timer() - te_time
                edge_scores = [edges for g_edges in all_graph_edges for edges in g_edges]
                edge_scores = pd.DataFrame(edge_scores, columns=['src', 'dest',  'score', 'snapshot', 'time', 'label', 'campaign'])
                result_file = self.args.dataset + '_d' + str(self.args.dimensions) + 'all_users.csv'
                self.calculate_performance_metrics(edge_scores, result_file,train_time,test_time,out_dir)
