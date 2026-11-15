from CTDNE.ctdne import CTDNE
from gensim.models import Word2Vec

from multiprocessing import Pool

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, Dropout
from tensorflow.keras.layers import GRU, Input, RepeatVector, TimeDistributed, Dense

import gc
import numpy as np
import pickle
from timeit import default_timer as timer
import os
import tensorflow as tf
os.environ["OMP_NUM_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

try:
    from gids_eval.process_monitor import maybe_profile_phase
except Exception:
    maybe_profile_phase = None

def short_term_embedding(args, node_list, idx, G):
    CTDNE_model = CTDNE(G, dimensions=args.dimensions, workers=4, walk_length=args.walklen,
                        num_walks=args.numwalk,
                        quiet=True)
    ctdne_model = CTDNE_model.fit(window=10, min_count=1,
                                  batch_words=4)
    node_embs = np.zeros((len(node_list), args.dimensions), dtype=np.float32)
    index_map = {n: idx for idx, n in enumerate(node_list)}
    for token in ctdne_model.wv.index_to_key:
        pos = index_map.get(token)
        if pos is not None:
            node_vec = ctdne_model.wv[token]
            if node_vec is not None:
                node_embs[pos] = np.array(node_vec)
    return node_embs

class PIKACHU:
    def __init__(self, args, node_list, node_map, graphs):
        self.args = args
        self.mask_val = 0.
        self.node_list = node_list
        self.node_map = node_map
        self.graphs = graphs
        self.model, self.encoder = None, None

    def get_model_lkbck(self):
        input = Input(batch_shape=(self.batch_size, self.args.lookback, self.args.dimensions))
        mask = Masking(mask_value=self.mask_val)(input)
        el1 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(mask)
        do_enc = Dropout(0.5)(el1)
        encoded = GRU(self.args.dimensions, return_sequences=False, stateful=False)(do_enc)

        rp = RepeatVector(self.args.lookback)(encoded)
        dl1 = GRU(self.args.dimensions, return_sequences=True, stateful=False)(rp)
        do_de = Dropout(0.5)(dl1)
        dl3 = GRU(self.args.dimensions * 4, return_sequences=True, stateful=False)(do_de)
        decoded = TimeDistributed(Dense(self.args.dimensions))(dl3)

        return Model(inputs=input, outputs=decoded), Model(inputs=input, outputs=encoded)

    def autoencoder_model(self, time_step, dim):
        input = Input(shape=(time_step, dim))
        mask = Masking(mask_value=self.mask_val)(input)
        el1 = GRU(64, return_sequences=True)(mask)
        do_enc = Dropout(0.3)(el1)
        encoded = GRU(128, return_sequences=False)(do_enc)

        rp = RepeatVector(time_step)(encoded)
        dl1 = GRU(128, return_sequences=True)(rp)
        do_de = Dropout(0.3)(dl1)
        dl3 = GRU(64, return_sequences=True)(do_de)
        decoded = TimeDistributed(Dense(dim))(dl3)

        return Model(inputs=input, outputs=decoded), Model(inputs=input, outputs=encoded)

    def long_term_embedding(self, short_term_embs):
        mask_val = 0.
        total_node, time_step, dim = short_term_embs.shape
        self.model, self.encoder = self.autoencoder_model(time_step, dim)
        print(self.model.summary())
        self.model.compile(optimizer='adam', loss='mse')
        print("\n\nTraining Long Term Model...")
        history = self.model.fit(short_term_embs, short_term_embs, epochs=self.args.epoch,
                                 verbose=1, validation_split=0.1)

        graph_embedding = self.model.predict(short_term_embs)
        return graph_embedding

    def learn_embedding(self, save_file):
        print("\nGenerating Short Term embedding...")
        total_cpu = os.cpu_count() or 1
        requested_workers = int(os.environ.get("EG1_PIKACHU_MAX_WORKERS", total_cpu))
        worker_count = max(1, min(total_cpu, requested_workers))
        print("\nNumber of CPU Available: ", total_cpu)
        print("Short Term Workers: ", worker_count)
        data_tuple = [(self.args, self.node_list, idx, G) for idx, G in enumerate(self.graphs)]
        s_time = timer()
        if maybe_profile_phase:
            with maybe_profile_phase("train"):
                with Pool(worker_count) as pool:
                    short_term_embs = pool.starmap(short_term_embedding, data_tuple)
        else:
            with Pool(worker_count) as pool:
                short_term_embs = pool.starmap(short_term_embedding, data_tuple)
        pool.close()
        print("\nShort Term embedding Completed...   [%s Sec.]" % (timer() - s_time))
        short_term_embs = np.array(short_term_embs)

        with open('weights/short_term' + save_file, 'wb') as f:
            pickle.dump(short_term_embs, f,protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()

        with open('weights/short_term' + save_file , 'rb') as f:
            short_term_embs = pickle.load(f)
        print("\n\nStarting Long Term Embedding...")
        short_term_embs = np.transpose(short_term_embs, (1, 0, 2))
        s_time = timer()
        if maybe_profile_phase:
            with maybe_profile_phase("train"):
                dynamic_embs = self.long_term_embedding(short_term_embs)
        else:
            dynamic_embs = self.long_term_embedding(short_term_embs)
        print("\nLong Term Embedding Completed...   [%s Sec.]" % (timer() - s_time))
        with open('weights/long_term' + save_file , 'wb') as f:
            pickle.dump(dynamic_embs, f,protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()
