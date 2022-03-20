encoders = (
    "clip",
    "efficientnetb7",
    "resnet50",
    "clip4clip",
    "r2plus1d",
    "swin",
    "yamnet",
    "yamnet-r2plus1d",
    "yamnet-clip",
    "yamnet-efficientnetb7",
    "yamnet-resnet50",
    "yamnet-clip4clip",
    "yamnet-swin",
)
models = (
    "lr",
    "xgboost",
    "mlp",
    "mlp_small",
    "mlp_large",
)
aggs = (
    "mean",
    "concat",
    "abs_diff",
    # "diff",
    # "mult",
)
mlp_size = (100, 100)
mlp_large_size = (500, 500)
mlp_small_size = (50, 50)

path_output = "experiment1.csv"
