import sys

from config import CONF_PARAMS, METRICS
from linkpred.utils import build_spark
from linkpred.features import extract_features


def main(in_path, out_path):
    sc = build_spark(**CONF_PARAMS)
    edge_list = sc.textFile(in_path)

    metrics_data = extract_features(sc, edge_list, METRICS)
    metrics_data.repartition(1).saveAsTextFile(out_path)


if __name__ == "__main__":
    in_path, out_path = sys.argv[1], sys.argv[2]
    # in_path, out_path = "data/test_data.txt", "data/similarity_metrics"
    main(in_path, out_path)
