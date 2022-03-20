from typing import NamedTuple

path_output = "experiment2.csv"


class MetricLearningParameters(NamedTuple):
    temperature: float
    mlp_dim1: int
    mlp_dim2: int
    mlp_lr: float
    weight_decay: float
    epochs: int


class TaskParameters(NamedTuple):
    frame: MetricLearningParameters
    motion: MetricLearningParameters


task_encoders = (
    ("frame", "clip"),
    ("frame", "clip4clip"),
    ("frame", "r2plus1d"),
    ("frame", "efficientnetb7"),
    ("frame", "resnet50"),
    ("frame", "swin"),
    ("frame", "yamnet"),
    ("motion", "r2plus1d"),
    ("motion", "swin"),
    ("motion", "clip4clip"),
    ("motion", "resnet50"),
    ("motion", "efficientnetb7"),
    ("motion", "clip"),
    ("motion", "yamnet"),
)

hyperparameters = TaskParameters(
    frame=MetricLearningParameters(
        temperature=0.007362,
        mlp_dim1=128,
        mlp_dim2=1_024,
        mlp_lr=0.003147,
        weight_decay=0.0001,
        epochs=300,
    ),
    motion=MetricLearningParameters(
        temperature=0.013412,
        mlp_dim1=256,
        mlp_dim2=1_024,
        mlp_lr=0.0004056,
        weight_decay=0.000454,
        epochs=100,
    ),
)

type_of_triplets = "hard"
batch_size_train = 256
batch_size_evaluation = 256
seed_step = 1000
