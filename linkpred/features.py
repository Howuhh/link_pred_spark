from .utils import prep_edges, compute_metrics
from .utils import to_csv, to_undirected

from .neighbors import get_neighbors, get_second_neighbors


def extract_features(sc, edge_list, metrics, top_n, delim):
    edge_list = edge_list.map(prep_edges).flatMap(to_undirected)

    neighbors = get_neighbors(edge_list)
    neighbors_cached = sc.broadcast(neighbors.collectAsMap()).value

    second_neighbors = neighbors.keys().map(
        lambda node: (node, get_second_neighbors(neighbors_cached, node))
    )

    metrics_data = second_neighbors.flatMap(
        lambda node: compute_metrics(neighbors_cached, metrics, node, top_n)
    )
    
    metrics_data_csv = metrics_data.map(
        lambda row: to_csv(row, delim)
    )

    return metrics_data_csv
