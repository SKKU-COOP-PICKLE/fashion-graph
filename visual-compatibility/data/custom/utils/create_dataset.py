"""
Create the dataset files for Polyvore from the raw data.
"""

import os
import random
import json
import scipy as sp
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import argparse
import pickle as pkl
import numpy as np
# from get_questions import get_questions
# from get_compatibility import get_compats
# from resample_fitb import resample_fitb
# from resample_compat import resample_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--phase', choices=['train', 'valid', 'test'], required=True)
args = parser.parse_args()

save_path = '../dataset/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(f'../jsons/{args.phase}.json') as f:
    json_data = json.load(f)

# load the features extracted with 'extract_features.py'
feat_pkl = os.path.join(save_path, f'imgs_featdict_{args.phase}.pkl')
if os.path.exists(feat_pkl):
    with open(feat_pkl, 'rb') as f:
        feat_dict = pkl.load(f)
else:
    raise FileNotFound('The extracted features file {} does not exist'.format(feat_pkl))


idx = 0
id2idx = {}
relations = {}
features = []
outfits = []
for fashion in json_data:
    item_ids = set(item['id'] for item in fashion['items'])
    item_ids = [id for id in item_ids if id in feat_dict]
    
    if len(item_ids) < 2:
        continue
    
    if args.phase == 'test':
        outfits.append((item_ids, 1))
    
    for i, id in enumerate(item_ids):
        if id not in relations:
            relations[id] = set()
            img_feats = feat_dict[id]
            features.append(img_feats)
            id2idx[id] = idx
            idx += 1

        if len(item_ids) > 1:
            relations[id].update(item_ids)
            relations[id].remove(id)

        
map_file = os.path.join(save_path, f"id2idx_{args.phase}.json")
with open(map_file, 'w') as f:
    json.dump(id2idx, f)

# create sparse matrix that will represent the adj matrix
sp_adj = lil_matrix((idx, idx))
features_mat = np.zeros((idx, 2048))

print('Filling the values of the sparse adj matrix')
for rel in relations:
    rel_list = relations[rel]
    from_idx = id2idx[rel]
    features_mat[from_idx] = features[from_idx]

    for related in rel_list:
        if related not in id2idx:
            continue
        to_idx = id2idx[related]

        sp_adj[from_idx, to_idx] = 1
        sp_adj[to_idx, from_idx] = 1 # because it is symmetric

print('Done!')

density = sp_adj.sum() / (sp_adj.shape[0] * sp_adj.shape[1])
print('Sparse density: {}'.format(density))

# now save the adj matrix
save_adj_file = os.path.join(save_path, f'adj_{args.phase}.npz')
sp_adj = sp_adj.tocsr()
save_npz(save_adj_file, sp_adj)

save_feat_file = os.path.join(save_path, f'features_{args.phase}.npz')
sp_feat = csr_matrix(features_mat)
save_npz(save_feat_file, sp_feat)


if args.phase == 'test':
    print(f'Unique outfit #: {len(outfits)}')
    random.seed(42)
    all_items = list(id2idx.keys())
    for i in range(len(outfits)):
        randoms = random.choices(all_items, k=random.randint(2, 5))
        outfits.append((randoms, 0))
    
    compat_file = os.path.join(
        save_path, 'compatibility_{args.phase}.json')
    print(compat_file)
    outfits = [[[id2idx[id] for id in item_ids], label]
               for item_ids, label in outfits]
    
    with open(compat_file, 'w') as f:
        json.dump(outfits, f)

    with open(os.path.join(save_path, 'compatibility_test.txt'), 'w') as f:
        for i in range(len(outfits)):
            indexes = [str(index) for index in outfits[i][0]]
            f.write(" ".join(indexes)+"\n")
            
# def create_test():
#     _outfits = get_compats()
#     outfits = []
#     for i in range(len(_outfits)): # for each outfit
#         ids = [id2idx[id] for id in _outfits[i][0] if id in id2idx]
#         if not ids or len(ids) == 1:
#             continue
#         outfits.append([ids, _outfits[i][1]])
        



# if args.phase == 'test':
#     create_test()
