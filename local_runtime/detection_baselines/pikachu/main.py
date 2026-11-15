import argparse
import os
import warnings
import pickle
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import DataUtils, GraphUtils
from anomaly_detection import AnomalyDetection
from pikachu import PIKACHU

warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    args = argparse.ArgumentParser("walk")
    args.add_argument('-k', "--dimensions", default = 200, type=int, help="Number of Dimension")
    args.add_argument('-ip', '--input', default = 'dataset/cic/cic_5.csv', help='Dataset filename')
    args.add_argument('-l', '--walklen', default = 500, type=int, help='Walk length')
    args.add_argument('-n', '--numwalk', default = 1, type=int, help='Walk length')
    args.add_argument('-w', '--trainwin', type=int, default = 98,
                        help='Context size for optimization. Default is 2.')
    args.add_argument('-e', '--epoch', default= 50, type=int,
                        help='Number of epochs (for GRU)')
    args.add_argument('-i', '--iter', default=10, type=int,
                      help='Number of iteration (for edge probability estimation)')
    args.add_argument('-r', '--alpha', default=0.001, type=float,
                      help='Learning rate for edge probability estimation')
    args.add_argument('-s', '--support', default=10, type=int,
                      help='Support Set (# of neighbor for edge probability estimation)')
    args.add_argument('-t', '--train', nargs='?', const=True, default = True, type=str2bool,
                      help='Is training?')
    args.add_argument('-o', '--output', type=str, default='results',
                         help='Output dir for anomalous edges')
    args.add_argument('-d', '--dataset', type=str, default='lanl',
                      help='Name of the dataset')
    args.add_argument('--metrics_csv', type=str, default=None,
                      help='Optional path to append evaluation metrics as CSV')
    args.add_argument('--gpu', type=str, default=None,
                      help='GPU id to use (e.g., \"0\"); leave empty/omit to let TensorFlow choose available GPU; set to \"-1\" to force CPU')
    return args.parse_args()

if __name__=="__main__":
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            print(f"Using GPUs: {gpus} (memory growth enabled)")
        else:
            print("No GPU detected, running on CPU.")
    except Exception as e:
        print(f"GPU setup warning: {e}")

    DST = args.output
    file = "sample"
    data_file = '_' + args.dataset + '_' + file + '.pickle'
    weights_dir = Path("weights")
    node_map_path = weights_dir / f"node_map{data_file}"
    graphs_path = weights_dir / f"graphs{data_file}"
    os.makedirs(DST, exist_ok=True)
    os.makedirs(os.path.join(DST, "learning"), exist_ok=True)
    os.makedirs(os.path.join(DST, "results"), exist_ok=True)
    reuse_graph_cache = node_map_path.exists() and graphs_path.exists()
    if args.train and not reuse_graph_cache:
        print("... Parsing Data ... \n")
        dp = DataUtils(data_folder=args.input)
        data_df, node_map = dp.get_data()
        with open(node_map_path, 'wb') as f:
           pickle.dump(node_map, f,protocol=pickle.HIGHEST_PROTOCOL)
        print("... Generating Graphs ... \n")
        g_util = GraphUtils(node_map)
        graphs = []
        for t in tqdm(data_df.snapshot.unique()):
            graphs.append(g_util.create_graph(data_df[data_df['snapshot'] == t]))
        with open(graphs_path, 'wb') as f:
           pickle.dump(graphs, f,protocol=pickle.HIGHEST_PROTOCOL)
    elif args.train and reuse_graph_cache:
        print("... Reusing Parsed Graph Cache ... \n")

    with open(node_map_path, 'rb') as f:
        node_map = pickle.load(f)

    with open(graphs_path, 'rb') as f:
        graphs = pickle.load(f)

    print("\nTotal Graphs: ", len(graphs))
    print(len(graphs[0].edges()))
    node_list = [str(v) for k, v in node_map.items()]
    print("\nTotal Nodes: ", len(node_map))

    print("... Starting Graph Embedding ... \n")

    weight_file = '_' + args.dataset + '_' + file + '_d' + str(args.dimensions) + '.pickle'
    print(" \n********** PARAM **********\n", args)
    print("Weight Files: ", weight_file)
    print("********************\n")
    short_term_path = weights_dir / f"short_term{weight_file}"
    long_term_path = weights_dir / f"long_term{weight_file}"
    if args.train and short_term_path.exists() and long_term_path.exists():
        print("... Reusing Embedding Cache (short_term + long_term weights exist) ... \n")
    elif args.train:
        pikachu = PIKACHU(args=args, node_list=node_list, node_map=node_map, graphs=graphs)
        pikachu.learn_embedding(weight_file)
    print("\n\n =====   Anomaly Detection ===== ")
    with open('weights/long_term' + weight_file, 'rb') as f:
        long_term_embs = pickle.load(f)
    long_term_embs = np.transpose(long_term_embs, (1, 0, 2))

    ad_long_term = AnomalyDetection(args, node_list, node_map, long_term_embs, idx = 0)
    param_file_name = '_' + args.dataset + '_' + file + '_d' + str(args.dimensions)
    ad_long_term.anomaly_detection(graphs, param_file='weights/param' + param_file_name, out_dir = DST)
    del ad_long_term
