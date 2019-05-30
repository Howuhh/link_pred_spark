import numpy as np


def common_neighbors_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)

    return len(common)


def adamic_adar_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)
    degrees = _common_degree(neighbors, common)

    score = np.sum(np.log1p(degrees))

    return np.round(score, 3)


def res_allocation_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)
    degrees = _common_degree(neighbors, common)

    score = np.sum(degrees)

    return score


def salton_index_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)
    deg1, deg2 = len(neighbors[n1]), len(neighbors[n2])
    score = len(common) / np.sqrt(deg1 * deg2)

    return score


def jaccard_index_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)
    union = _union_neighbors(neighbors, n1, n2)
    score = len(common) / len(union)

    return score


def sorensen_index_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)
    deg1, deg2 = len(neighbors[n1]), len(neighbors[n2])
    score = (2 * len(common)) / (deg1 + deg2)

    return score


def hub_promoted_index_score(neighbors, n1, n2):
    common = _common_degree(neighbors, n1, n2)
    deg1, deg2 = len(neighbors[n1]), len(neighbors[n2])
    score = len(common) / np.minimum(deg1, deg2)

    return score


def lh_newman_index_score(neighbors, n1, n2):
    common = _common_neighbors(neighbors, n1, n2)
    deg1, deg2 = len(neighbors[n1]), len(neighbors[n2])
    score = len(common) / (deg1 * deg2)

    return score


def preferential_attachment_score(neighbors, n1, n2):
    score = len(neighbors[n1]) * len(neighbors[n2])

    return score


def _common_neighbors(neighbors, n1, n2):
    common = set(neighbors[n1]) & set(neighbors[n2])

    return list(common)


def _common_degree(neighbors, common):
    degrees = [len(neighbors[node]) for node in common]

    return degrees


def _union_neighbors(neighbors, n1, n2):
    union = set(neighbors[n1]) | set(neighbors[n2])

    return list(union)
