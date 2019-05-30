import findspark
from . import metrics

findspark.init()
from pyspark import SparkConf, SparkContext


def build_spark(**kwargs):
    сonf = SparkConf()
    for param in kwargs:
        assert isinstance(param, str), "must be string type"
        try:
            сonf.set(param, kwargs[param])
        except KeyError:
            raise KeyError(f"Unknown configuration parameter: {param}")
    sc = SparkContext(conf=сonf)
    return sc


def prep_edges(row):
    row_split = row.split(" ")
    nodes = [int(node) for node in row_split]

    return nodes


def to_undirected(node):
    double = ((node[0], node[1]), (node[1], node[0]))

    return double


def get_neighbors(edge_list):
    neighbors = edge_list.groupByKey().map(
        lambda node: [node[0], node[1].data]
    )

    return neighbors


def get_second_neighbors(neighbors, node):
    node_neighbors = neighbors[node]
    visited = []

    for node_ in node_neighbors:
        visited.extend(neighbors[node_])

    second_neighbors = set(visited) - set(node_neighbors + [node])

    return list(second_neighbors)


def compute_metrics(neighbors, metrics_, node, top_n):
    node, cands = node
    scores = []

    for cand in cands:
        node_scores = [node, cand]
        for metric in metrics_:
            try:
                scorer = getattr(metrics, metric)
                score = scorer(neighbors, node, cand)
                node_scores.append(score)
            except AttributeError:
                raise AttributeError("Unknown metric: {metric}")
        scores.append(node_scores)

    scores = sorted(scores, key=lambda scores: scores[1])

    return scores[:top_n]
