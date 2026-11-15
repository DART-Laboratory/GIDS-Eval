from collections import defaultdict
import numpy as np
import gensim
from joblib import Parallel, delayed
from tqdm import tqdm
from .parallel import parallel_generate_walks

class CTDNE:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    NEIGHBORS_TIME_KEY = 'neighbors_time'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'
    GRAPH_ID = 'gid'

    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, use_linear=True, half_life=1):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.use_linear = use_linear
        self.half_life = half_life

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.d_graph = self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        d_graph = defaultdict(dict)
        first_travel_done = set()

        nodes_generator = self.graph.nodes() if self.quiet\
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            if self.FIRST_TRAVEL_KEY not in d_graph[source]:
                d_graph[source][self.FIRST_TRAVEL_KEY] = dict()

            for current_node in self.graph.neighbors(source):
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

                neighbor2times = {}
                for neighbor in d_neighbors:
                    neighbor2times[neighbor] = []
                    if 'time' in self.graph[current_node][neighbor]:
                        neighbor2times[neighbor].append(self.graph[current_node][neighbor]['time'])
                    else:
                        for att in list(self.graph[current_node][neighbor].values()):
                            if 'time' not in att:
                                raise ('no time attribute')
                            neighbor2times[neighbor].append(att['time'])
                d_graph[current_node][self.NEIGHBORS_TIME_KEY] = neighbor2times

        return d_graph

    def _generate_walks(self):

        flatten = lambda l: [item for sublist in l for item in sublist]

        num_walks_lists = np.array_split(range(self.num_walks), self.workers)
        walk_results = Parallel(n_jobs=self.workers)(delayed(parallel_generate_walks)(self.d_graph,
                                                                                      self.walk_length,
                                                                                      len(num_walks),
                                                                                      idx,
                                                                                      self.sampling_strategy,
                                                                                      self.NUM_WALKS_KEY,
                                                                                      self.WALK_LENGTH_KEY,
                                                                                      self.NEIGHBORS_KEY,
                                                                                      self.NEIGHBORS_TIME_KEY,
                                                                                      self.PROBABILITIES_KEY,
                                                                                      self.FIRST_TRAVEL_KEY,
                                                                                      self.quiet,
                                                                                      self.use_linear,
                                                                                      self.half_life) for
                                                     idx, num_walks
                                                     in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def fit(self, **skip_gram_params):

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        version = getattr(gensim, "__version__", "4.0.0").split(".")
        try:
            major = int(version[0])
        except Exception:
            major = 4

        if major >= 4:
            if "vector_size" not in skip_gram_params:
                if "size" in skip_gram_params:
                    skip_gram_params["vector_size"] = skip_gram_params.pop("size")
                else:
                    skip_gram_params["vector_size"] = self.dimensions
            skip_gram_params.pop("size", None)
        else:
            if "size" not in skip_gram_params:
                if "vector_size" in skip_gram_params:
                    skip_gram_params["size"] = skip_gram_params.pop("vector_size")
                else:
                    skip_gram_params["size"] = self.dimensions
            skip_gram_params.pop("vector_size", None)

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)
