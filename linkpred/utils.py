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


def to_csv(row, delim):
    assert isinstance(delim, str), "delimiter should be string"
    csv_row = f"{delim}".join(str(i) for i in row)

    return csv_row


def compute_metrics(neighbors, metrics_, node, top_n):
    node, cands = node
    scores = []

    for cand in cands:
        node_scores = [node, cand]
        for metric in metrics_:
            try:
                scorer = getattr(metrics, metric)
            except AttributeError:
                raise AttributeError("Unknown metric: {metric}")
            # compute similarity
            score = scorer(neighbors, node, cand)
            node_scores.append(score)
        scores.append(node_scores)

    # sort by first metric
    scores = sorted(scores, key=lambda scores: scores[2])

    return scores[:top_n]
