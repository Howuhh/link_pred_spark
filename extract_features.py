import click

from config import CONF_PARAMS, METRICS
from linkpred.utils import build_spark
from linkpred.features import extract_features


@click.command()
@click.argument('in_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path())
@click.option('--top_n', default=40, help="""The number of most similar candidates to leave for final 
                                            dataset for each node. All other candidates will be filtered.""")
@click.option('--csv_delim', default=",", help="СSV file separator, should be string.")
def main(in_path, out_path, top_n, csv_delim):
    """
    This module allows you to calculate similarity
    metrics based on local information for the nodes
    of the graph, ranking by the number of the most
    similar ones, and easily add new metrics.

    Friends of friends are selected as candidates for each node.
    The final dataset with metrics can be used to train machine
    learning models to predict edges between nodes
    which are likely to appear in the future.

    You should specify IN_PATH for input graph data in edgelist format 
    and OUT_PATH along which the final dataset with metrics will be saved. 
    Metrics and Spark configuration settings should be specified in config.py.
    """
    sc = build_spark(**CONF_PARAMS)
    edge_list = sc.textFile(in_path)

    metrics_data = extract_features(sc, edge_list, METRICS, top_n, csv_delim)
    metrics_data.repartition(1).saveAsTextFile(out_path)


if __name__ == "__main__":
    # in_path, out_path = "data/test_data.txt", "data/similarity_metrics"
    main()
