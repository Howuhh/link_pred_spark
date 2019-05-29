
# docstring here
CONF_PARAMS = {
    "spark.master": "local[*]",
    "spark.app.name": "linkpred_spark",
}

# docstring here
METRICS = [
    "common_neighbors_score",
    "adamic_adar_score",
    "res_allocation_score",
]
