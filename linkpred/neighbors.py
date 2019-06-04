

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

    second_neighbors = _diff_neighbors(visited, node_neighbors + [node])

    return list(second_neighbors)


def _common_neighbors(neighbors, n1, n2):
    common = set(neighbors[n1]) & set(neighbors[n2])

    return list(common)


def _common_degree(neighbors, common):
    degrees = [len(neighbors[node]) for node in common]

    return degrees


def _union_neighbors(neighbors, n1, n2):
    union = set(neighbors[n1]) | set(neighbors[n2])

    return list(union)


def _diff_neighbors(neighbors1, neighbors2):
    diff = set(neighbors1) - set(neighbors2)

    return diff
