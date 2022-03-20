from sklearn import metrics

random_state = 0
cache = True  # cache experiment results locally
tasks = ("frame", "motion")
include_metrics = ("train", "validation", "test")
metric_fn = metrics.average_precision_score
bootstrap_cnt = 5
# overrides the default embedding path if not None
path_embeddings = None

# table
round_to = 3

# embeddings dimensions
embedding_dims = {
    "clip": 512,
    "clip4clip": 512,
    "yamnet": 1_024,
    "efficientnetb7": 2_560,
    "resnet50": 2_048,
    "r2plus1d": 512,
    "swin": 1_024,
    "yamnet-r2plus1d": 1_536,
    "yamnet-clip": 1_536,
    "yamnet-clip4clip": 1_536,
    "yamnet-efficientnetb7": 3_584,
    "yamnet-resnet50": 3_072,
    "yamnet-swin": 2_048,
}
