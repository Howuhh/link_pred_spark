# Similarity between nodes

This module allows you to calculate similarity metrics based on local information for the nodes of the graph, ranking by the number of the most similar ones, and easily add new metrics. 

Friends of friends are selected as candidates for each node. The final dataset with metrics can be used to train machine learning models to predict edges between nodes which are likely to appear in the future.

For additional info about metrics see [predicting missing links via local information](http://image.sciencenet.cn/olddata/kexue.com.cn/upload/blog/file/2009/10/2009102822251329127.pdf) paper.

# How to use

The input graph data should be in an edge list format.  In order to compute the metrics, you need to specify the path to the source data and the path along which the final dataset will be saved.

```console
Usage: extract_features.py [OPTIONS] IN_PATH OUT_PATH

  This module allows you to calculate similarity metrics based on local
  information for the nodes of the graph.

  You should specify IN_PATH for input graph data in edgelist format  and
  OUT_PATH along which the final dataset with metrics will be saved.
  Metrics and Spark configuration settings should be specified in config.py.

Options:
  --top_n INTEGER   The number of most similar candidates to leave for final
                    dataset for each node. All other candidates will be
                    filtered.
  --csv_delim TEXT  СSV file separator, should be string.
  --help            Show this message and exit.
```

# How to add new metrics
Each metric function must take a dictionary of the form of `{node_id: [list of neighbors]}`, as well as a pair of nodes for comparison.

**Example**:    
The use case of how to add new metrics is demonstrated.
We will add a metric based on resource allocation. To do this, let's define the function according to the arguments' requirements.

```python
# linkpred/metrics.py
import numpy as np

def res_allocation_score(neighbors, node1, node2):
    """
    Similarity measure based on resource allocation. 
    For a detailed explanation see Tao Zhou, Linyuan Lu ̈ 
    and Yi-Cheng Zhang paper: https://arxiv.org/abs/0901.0553
    
    Parameters
    ----------
    neighbors: dict
        Dictionary in format {node_id: [list_of_neighbors]},
    node1: int
        Node id
    node2: int
        Node id
    Returns
    -------
    score: int
        Nodes similarity score.
    """
    common = _common_neighbors(neighbors, node1, node2)
    degrees = _common_degree(neighbors, common)

    score = np.sum(degrees)

    return score
```
We will also use the helper functions from metrics module to calculate the common neighbors for the two nodes and the degree of common neighbors.

The only thing left to do is to add this to the config, which in this case is just a list or dictionary. 

```python
# config.py
METRICS = [
    "common_neighbors_score",
    "adamic_adar_score",
    "res_allocation_score",
]
```
That's all it is. The order in the final dataset is maintained, so this metric will be the last column.

# Spark config

To configure configuration settings for spark, you can also change config. For example, let's change the number of cores to two.

```python
# config.py
CONF_PARAMS = {
    "spark.master": "local[2]",
    "spark.app.name": "linkpred_spark",
}
```

# Available metrics

By default, these metrics are available: 

* Common Neighbours 
* Salton Index  
* Jaccard Index 
* Sørensen Index    
* Hub Promoted Index
* Leicht-Holme-Newman Index
* Preferential Attachment
* Adamic-Adar Index
* Resource Allocation

# P.S.

[A more optimized implementation](https://github.com/marnikitta/missing-links) in which you do not have to return the data to the driver to calculate metrics for pairs. This is achieved by generating triplets and calculating partial sums.
