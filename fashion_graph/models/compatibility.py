import os
from typing import *
import json
import numpy as np
from itertools import combinations

from .base import GraphBase


class GraphCompat(GraphBase):
    """
    Calculate compatibilities of items
    """
    def __init__(self, model_dir: str, model_config: object = None):
        """
        :param model_dir: The directory path of saved `visual-compatibility` model.
            (It must have `best_epoch.ckpt` and `results.json`)
        :param model_config: tf.ConfigProto()
        """
        super().__init__(model_dir, model_config)

    def predict_no_adj(self, features: np.ndarray) -> float:
        """
        Calculate compatibility of items without adjacency matrix of items
    
        :param features: The `np.ndarray` feature vector array of items. The size of feature vector must be same as model's `input_dim`
        
        return: The compatibility score
        """

        # make questions for compatibility
        questions = list(combinations(range(len(features)), r=2))
        # predict
        predictions = super().predict_no_adj(questions=questions, features=features)
        # prediction score
        score = np.mean(predictions)
        return score

    def predict(self, query_index: int, adj: np.ndarray, features: np.ndarray, k: int) -> float:
        """
        Calculate compatibility of items with adjacency matrix of items
        More accurate than 'without adjacency matrix'
        
        :param query_index: The index of item you want to recommend
        :param adj: The `np.ndarray` adjacency matrix 
        :param features: The `np.ndarray` feature vector array of items. The size of feature vector must be same as model's `input_dim`
        :param k:  Maximum number of graph neighbors to search (BFS)
        
        return: The compatibility score
        """
        assert adj.shape[0] == features.shape[0], \
            "the length of adjacency matrix should be same with the length of features"
        assert k > 0, "k should be positive"

        # make questions for compatibility
        questions = list(combinations(range(len(features)), r=2))
        # predict
        predictions = super().predict(questions=questions, adj=adj, features=features, k=k)
        # prediction score
        score = np.mean(predictions)
        return score
