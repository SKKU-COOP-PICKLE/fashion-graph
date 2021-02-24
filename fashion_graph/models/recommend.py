import os
from typing import *
import json
import numpy as np
from itertools import product

from .base import GraphBase


class GraphRecommender(GraphBase):
    """
    Recommend items
    """
    def __init__(self, model_dir: str, model_config: object):
        """
        :param model_dir: The directory path of saved `visual-compatibility` model.
            (It must have `best_epoch.ckpt` and `results.json`)
        :param model_config: tf.ConfigProto()
        """
        super().__init__(model_dir, model_config)

    def predict_no_adj(self, query_index: int, features: np.ndarray, topk: int = 10, filter_indexes: List[int] = []) -> List[Tuple[int, float]]:
        """
        Recommend indexes of top-K items without adjacency matrix of items
        
        :param query_index: The index of item you want to recommend
        :param filter_indexes: The List of indexes you want to exclude from recommendations
        :param features: The `np.ndarray` feature vector array of items. The size of feature vector must be same as model's `input_dim`
        :param topk: The number of recommended items
        
        return: The `list` of `tuple(item index, score)` 
        """
        
        # filter candidate indexes
        candidate_indexes = range(len(features))
        candidate_indexes = list(filter(lambda index: index not in filter_indexes, candidate_indexes))
        
        # make questions for recommendation
        questions = list(product([query_index], candidate_indexes))
        # predict
        scores = np.array(super().predict_no_adj(questions=questions, features=features))
        
        assert len(candidate_indexes) == len(scores)
        
        # argsort the prediction score (desc)
        topk_scores_indexes = scores.argsort()[::-1][:topk]
        topk_indexes = np.array(candidate_indexes)[topk_scores_indexes]
        
        return [(index, score) for index, score in zip(topk_indexes, scores[topk_scores_indexes])]

    def predict(self, query_index: int, adj: np.ndarray, features: np.ndarray, k: int, topk: int = 10, filter_indexes: List[int] = []) -> List[Tuple[int, float]]:
        """
        Recommend indexes of top-K items with adjacency matrix of items.
        More accurate than 'without adjacency matrix'
        
        :param query_index: The index of item you want to recommend
        :param adj: The `np.ndarray` adjacency matrix 
        :param features: The `np.ndarray` feature vector array of items. The size of feature vector must be same as model's `input_dim`
        :param k:  Maximum number of graph neighbors to search (BFS)
        :param topk: The number of recommended items
        
        return: The `list` of `tuple(item index, score)`
        """
        assert adj.shape[0] == features.shape[0], \
            "the length of adjacency matrix should be same with the count of features"
        assert k > 0, "k should be positive"
        
        # filter candidate indexes
        candidate_indexes = range(len(features))
        candidate_indexes = list(filter(lambda index: index not in filter_indexes, candidate_indexes))
        
        # make questions for recommendation
        questions = list(product([query_index], candidate_indexes))
        # predict
        scores = np.array(super().predict(questions=questions, adj=adj, features=features, k=k))
        
        assert len(candidate_indexes) == len(scores)
        
        # argsort the prediction score (desc)
        topk_scores_indexes = scores.argsort()[::-1][:topk]
        topk_indexes = np.array(candidate_indexes)[topk_scores_indexes]
        
        return [(index, score) for index, score in zip(topk_indexes, scores[topk_scores_indexes])]
