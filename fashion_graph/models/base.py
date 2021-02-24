import os
import json
from typing import *
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from .gcn.model import CompatibilityGAE
from .gcn.utils import Graph
from .gcn.utils import get_degree_supports, sparse_to_tuple, normalize_nonsym_adj, construct_feed_dict


def get_norm_supports(adj, degree):
    supports = get_degree_supports(
        adj, degree, adj_self_con=True, verbose=False)
    for i in range(1, len(supports)):
        supports[i] = normalize_nonsym_adj(supports[i])
    return [sparse_to_tuple(sup) for sup in supports]


class GraphBase(object):
    def __init__(self, model_dir: str):
        """
        :param model_dir: The directory path of saved `visual-compatibility` model.
            It must have `best_epoch.ckpt` and `results.json`)
        """
        # visual-compatibility model
        model_path = f"{model_dir}/best_epoch.ckpt"
        with open(f"{model_dir}/results.json") as f:
            config = json.load(f)

        self.degree = config['degree']
        self.placeholders = {
            'row_indices': tf.placeholder(tf.int32, shape=(None,)),
            'col_indices': tf.placeholder(tf.int32, shape=(None,)),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'weight_decay': tf.placeholder_with_default(0., shape=()),
            'is_train': tf.placeholder_with_default(False, shape=()),
            'support': [tf.sparse_placeholder(tf.float32, shape=(None, None)) for sup in range(self.degree+1)],
            'node_features': tf.placeholder(tf.float32, shape=(None, None)),
            'labels': tf.placeholder(tf.float32, shape=(None,))
        }

        self.input_dim = 2048  # default (resnet50)
        self.model = CompatibilityGAE(
            self.placeholders,
            input_dim=self.input_dim,
            num_classes=2,
            num_support=self.degree + 1,
            hidden=config['hidden'],
            learning_rate=config['learning_rate'],
            logging=False,
            batch_norm=config['batch_norm'])
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.saver.restore(self.session, model_path)

    def predict_no_adj(self, questions: List[Tuple[int, int]], features: np.ndarray) -> List[Tuple[int, float]]:
        """
        Predict compatibilities without adjacency matrix
        
        :param questions: The pairs of indexes that want to predict compatibilities
        :param features: The `np.ndarray` feature vector array of items. The size of feature vector must be same as model's `input_dim`
        
        return: The `list` of `tuple(item index, score)`
        """
        assert features.shape[1] == self.model.input_dim

        query_r = [q[0] for q in questions]
        query_c = [q[1] for q in questions]

        n_items = features.shape[0]
        adj = sp.csr_matrix((n_items, n_items))

        feed_dict = construct_feed_dict(self.placeholders, features, get_norm_supports(adj, self.degree),
                                        [], query_r, query_c, 0., is_train=False)

        output = self.session.run(tf.nn.sigmoid(self.model.outputs), feed_dict=feed_dict)

        query_indexes = [query_c[index] for index in output]
        scores = output[query_indexes]

        return [(index, score) for index, score in zip(query_indexes, scores)]

    def predict(self, questions: List[Tuple[str, str]], adj: np.ndarray, features: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Predict compatibilities with adjacency matrix
        
        :param questions: The pairs of indexes that want to predict compatibilities
        :param adj: The `np.ndarray` adjacency matrix 
        :param features: The `np.ndarray` feature vector array of items. The size of feature vector must be same as model's `input_dim`
        :param k: Maximum number of graph neighbors to search (BFS)
        
        return: The `list` of `tuple(item index, score)`
        """
        assert adj.shape[0] == features.shape[0], \
            "the length of adjacency matrix should be same with the length of features"
        assert k > 0, "k should be positive"

        query_r = [q[0] for q in questions]
        query_c = [q[1] for q in questions]

        new_adj = sp.csr_matrix(adj.shape)

        graph = Graph(adj.copy().tolil())
        for r, c in zip(query_r, query_c):
            graph.remove_edge(c, r)
            graph.remove_edge(r, c)

        graph.adj = graph.adj.tocsr()
        graph.adj.eliminate_zeros()

        nodes_to_expand = np.unique(query_r + query_c)
        for node in nodes_to_expand:
            edges = graph.run_K_BFS(node, k)
            for edge in edges:
                u, v = edge
                new_adj[u, v] = 1
                new_adj[v, u] = 1

        new_adj = new_adj.tocsr()

        feed_dict = construct_feed_dict(self.placeholders, features, get_norm_supports(adj, self.degree),
                                        [], query_r, query_c, 0., is_train=False)

        output = self.session.run(tf.nn.sigmoid(self.model.outputs), feed_dict=feed_dict)

        query_indexes = [query_c[index] for index in output]
        scores = output[query_indexes]

        return [(index, float(score)) for index, score in zip(query_indexes, scores)]

    def build_adj(self, relations: List[Tuple[int, int]]):
        """
        Build adj matrix based on relations.
        Tuple in `relations` should be (item index, item index).
        Relations should be considered undirected.
        
        ex. [(0, 1), (1, 2)] => [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

        returns `scipy.sparse.csr_matrix`
        """
        size = max(sum(relations, ())) + 1
        adj = sp.csr_matrix((size, size))
        for u, v in relations:
            adj[u, v] = 1
            adj[v, u] = 1
        
        return adj.tocsr()
