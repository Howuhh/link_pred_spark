
# spark context configuration
CONF_PARAMS = {
    "spark.master": "local[*]",
    "spark.app.name": "linkpred_spark",
}

# metrics to compute
METRICS = [
    "common_neighbors_score",
    "adamic_adar_score",
    "res_allocation_score",
]
