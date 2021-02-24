# fashion-graph

Recommends best fashion items by calculating fashion compatibility.
Based on GCN(Graph Convolutional Network) model in [visual-compatibility](https://github.com/gcucurull/visual-compatibility).

Two methods are possible

- Recommends fashion items when you input one item

- Calculates compatibilities of input fashion items

## Train model

Check `visual-compatibility/`

## Setup
> Python 3.7

    python setup.py install

## Usage

You should have trained model.

### Extract image features

```python
from feature import FeatureExtractor

extractor = FeatureExtractor(device=device)
feature = extractor.get_feature(image)

```

### Build adjacency matrix

```python
adj = (GraphRecommender, GraphCompat).build_adj(relations)
```

### Recommend

```python
from fashion_graph import GraphRecommender

recommender = GraphRecommender(model_dir)
recommender.predict(query_index, adj, features, k, topk, filter_indexes)

# if you don't have any graph information, use predict_no_adj
recommender.predict_no_adj(query_index, features, topk, filter_indexes)
```

### Calculate Compatibility

```python
from fashion_graph import GraphCompat

compat = GraphCompat(model_dir)
compat.predict(adj, features, k, topk)

# if you don't have any graph information, use predict_no_adj
compat.predict_no_adj(features, topk)
```

## Example

    python example.py --k K --topk TOPK --image_path IMAGE_PATH
