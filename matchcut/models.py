import numpy as np
from dataclasses import dataclass
import functools
import random
from typing import Sequence, Tuple

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning import losses, miners
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import torch
from torch import nn
import torch.optim as optim
from scipy.spatial.distance import cosine
from xgboost import XGBClassifier

from .config import common, experiment1
from .data import (
    Dataset,
    DataLoader,
    Embedding,
    FeatureMatrix,
    Label,
    MetricLearningDataset,
)


@dataclass(frozen=True)
class Model:
    model_name: str
    random_state: int = common.random_state

    @property
    @functools.lru_cache()
    def model(self) -> Pipeline:
        choices = dict(
            lr=LogisticRegression,
            xgboost=XGBClassifier,
            mlp=functools.partial(
                MLPClassifier, hidden_layer_sizes=experiment1.mlp_size
            ),
            mlp_small=functools.partial(
                MLPClassifier, hidden_layer_sizes=experiment1.mlp_small_size
            ),
            mlp_large=functools.partial(
                MLPClassifier, hidden_layer_sizes=experiment1.mlp_large_size
            ),
        )
        clf = choices[self.model_name](random_state=self.random_state)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", clf),
            ]
        )

    def fit(self, x: FeatureMatrix, y: Sequence[Label]) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: FeatureMatrix) -> Tuple[float]:
        return tuple(self.model.predict_proba(x)[:, 1])


class MLP(nn.Module):
    def __init__(self, input_size: int, dim1: int, dim2: int):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, dim1),
            nn.LeakyReLU(),
            nn.Linear(dim1, dim2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass(frozen=False)
class MetricLearning:
    task: str
    encoder: str
    temperature: float
    type_of_triplets: str
    mlp_input: int
    mlp_dim1: int
    mlp_dim2: int
    mlp_lr: float
    weight_decay: float
    epochs: int
    batch_size_train: int
    batch_size_eval: int
    random_state: int
    seed_step: int

    def __post_init__(self):
        # data
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(False)
        self.datasets = {
            split: Dataset(
                split=split, task=self.task, encoder_name=self.encoder, agg_name=None
            )
            for split in ("train", "validation", "test")
        }

        self.ml_dataset_validation = MetricLearningDataset.by_args(
            self.task, "validation", self.encoder
        )
        self.ml_dataset_test = MetricLearningDataset.by_args(
            self.task, "test", self.encoder
        )
        # metric learning
        self.model = MLP(self.mlp_input, self.mlp_dim1, self.mlp_dim2).to(self.device)
        self.opt = optim.Adam(
            self.model.parameters(),
            lr=self.mlp_lr,
            weight_decay=self.weight_decay,
        )
        distance = CosineSimilarity()
        self.loss_fn = losses.NTXentLoss(
            temperature=self.temperature, distance=distance
        )
        self.mining_func = miners.TripletMarginMiner(
            distance=distance,
            type_of_triplets=self.type_of_triplets,
        )

    def train_dataloader(self, random_state: int) -> DataLoader:
        # TODO: add util so we can shuffle the same dataset instead of recomputing per epoch
        ds = MetricLearningDataset.by_args(
            self.task, "train", self.encoder, random_state=random_state
        )
        return ds.dataloader(batch_size=self.batch_size_train)

    @property
    def dataset_train(self) -> Dataset:
        return self.datasets["train"]

    @property
    def dataset_validation(self) -> Dataset:
        return self.datasets["validation"]

    @property
    def dataset_test(self) -> Dataset:
        return self.datasets["test"]

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def cos(x: Embedding, y: Embedding) -> float:
        return 1 - cosine(x, y)

    def train(self) -> None:
        self.model.train()

        loss_best = float("inf")
        for epoch in range(1, self.epochs + 1):
            loss_total = 0
            dl = self.train_dataloader(
                random_state=epoch + self.random_state * self.seed_step
            )
            for data, indices in dl:
                data, indices = data.to(self.device), indices.to(self.device)
                self.opt.zero_grad()
                embeddings = self.model(data)
                indices_tuple = self.mining_func(embeddings, indices)
                loss = self.loss_fn(embeddings, indices, indices_tuple)
                loss.backward()
                self.opt.step()
                loss_total += loss.item()
            loss_best = min(loss_best, loss_total)
            print(
                f"epoch {epoch:4d}, loss: {loss_total:.4f}, loss best: {loss_best:.4f}"
            )

    @property
    def probs_train(self) -> Tuple[float]:
        ml_ds = MetricLearningDataset(
            self.datasets["train"],
            MetricLearningDataset.get_ds_train(self.task, self.encoder),
        )
        return self.eval(ml_ds)

    @property
    def probs_validation(self) -> Tuple[float]:
        return self.eval(self.ml_dataset_validation)

    @property
    def probs_test(self) -> Tuple[float]:
        return self.eval(self.ml_dataset_test)

    def eval(self, ds: MetricLearningDataset) -> Tuple[float]:
        self.model.eval()
        with torch.no_grad():
            embs_list = [
                self.model(data.to(self.device)).cpu().detach().numpy()
                for data, _ in ds.dataloader(self.batch_size_eval)
            ]

        return tuple(
            self.cos(
                embs[pair_beg * 2],
                embs[pair_beg * 2 + 1],
            )
            for embs in embs_list
            for pair_beg in range(len(embs) // 2)
        )
