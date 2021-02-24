import os
import json
import torch
import argparse
import numpy as np
import PIL
from PIL import Image
import scipy.sparse as sp

from fashion_graph import GraphRecommender, FeatureExtractor
from fashion_graph import expand_csr_adj


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', required=True, type=str)
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--topk', default=10, type=int)
args = parser.parse_args()


MODEL_DIR = 'data/models/best'
DATA_DIR = 'data/all'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
recommender = GraphRecommender(model_dir=MODEL_DIR)
extractor = FeatureExtractor(device=device)

# load precomputed data
adj = sp.load_npz(f"{DATA_DIR}/adj.npz").astype(np.int32)
features = np.array(sp.load_npz(f"{DATA_DIR}/features.npz").todense())
with open(f"{DATA_DIR}/idx2id.json") as f:
    idx2id = json.load(f)

# read image
image = Image.open(args.image_path)
image = image.convert('RGB')
image = np.array(image)

new_index = features.shape[0]

# append feature
feature = extractor.get_feature(image)
_features = np.vstack((features, feature))

if args.k == 0:
    output = recommender.predict_no_adj(
        query_index=new_index, features=_features, topk=args.topk)
else:
    _adj = expand_csr_adj(adj, count=1)
    output = recommender.predict(
        query_index=new_index, adj=_adj, features=_features, k=args.k, topk=args.topk)

for index, score in output:
    print(idx2id[index], score)
