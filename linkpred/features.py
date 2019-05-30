from .utils import to_undirected, prep_edges, compute_metrics
from .utils import get_neighbors, get_second_neighbors


def extract_features(sc, edge_list, metrics, top_n=40):
    edge_list = edge_list.map(prep_edges).flatMap(to_undirected)

    neighbors = get_neighbors(edge_list)
    neighbors_cached = sc.broadcast(neighbors.collectAsMap()).value

    second_neighbors = neighbors.keys().map(
        lambda node: (node, get_second_neighbors(neighbors_cached, node))
    )

    metrics_data = second_neighbors.flatMap(
        lambda node: compute_metrics(neighbors_cached, metrics, node, top_n)
    )

    return metrics_data
