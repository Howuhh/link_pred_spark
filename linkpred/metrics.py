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


def _common_neighbors(neighbors, n1, n2):
    common = set(neighbors[n1]) & set(neighbors[n2])

    return list(common)


def _common_degree(neighbors, common):
    degrees = [len(neighbors[node]) for node in common]

    return degrees
