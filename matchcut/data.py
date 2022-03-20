import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset as TorchDataset

from .config import common

Embedding = FeatureMatrix = np.ndarray
Label = bool


RankingMetricFn = Callable[[Sequence[Label], Sequence[float]], float]


@dataclass(frozen=True)
class SourceInfo:
    sources_and_scores: Tuple[Tuple[str, float]]

    def from_source(self, source: str) -> bool:
        return any(src == source for src, _ in self.sources_and_scores)

    def get_score(self, source: str) -> float:
        return next(score for src, score in self.sources_and_scores if src == source)


@dataclass(frozen=True)
class Pair:
    imdb_id: str
    shot1_idx: int
    shot2_idx: int
    label: Label
    source_info: Optional[SourceInfo]


@dataclass(frozen=True)
class Dataset:
    split: str
    task: str
    encoder_name: Optional[str]
    agg_name: Optional[str] = None
    source: Optional[str] = None
    path_embeddings: Optional[str] = common.path_embeddings
    positive_only: bool = False
    random_state: Optional[int] = None

    @property
    def path_data(self) -> Path:
        here = Path(__file__).parent.resolve()
        one_level_up = here.parent
        return one_level_up / Path("data")

    @staticmethod
    def _fix_emb(imdb_id: str, name: str, emb: Embedding, idx: int) -> Embedding:
        dim = common.embedding_dims[name]
        # make sure that embeddings are consistently shaped for all shots
        valid = emb is not None and isinstance(emb, Iterable) and len(emb) == dim
        if not valid:
            emb_len = len(emb) if isinstance(emb, Iterable) else 0
            raise ValueError(
                f"invalid dims for {imdb_id} encoder {name} at idx {idx}. expected: {dim}, got: {emb_len}"
            )
        return emb

    @property
    def path_emb(self) -> Path:
        return Path(self.path_embeddings or self.path_data / Path(f"embeddings"))

    @functools.lru_cache()
    def _get_embeddings(self, encoder_name: str, imdb_id: str) -> Dict[int, Embedding]:
        path = self.path_emb / Path(f"{encoder_name}-{imdb_id}.json")
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception as e:
                raise ValueError(f"error reading embedding file {path} {e}")
        return {
            int(key): self._fix_emb(imdb_id, encoder_name, emb, int(key))
            for key, emb in data.items()
        }

    def encoder(self, imdb_id: str, idx: int) -> Embedding:
        embs = self._get_embeddings(self.encoder_name, imdb_id)
        return embs[idx]

    @property
    def agg(self) -> Callable[[Embedding, Embedding], Embedding]:
        aggs = dict(
            mean=lambda x, y: (np.array(x) + np.array(y)) / 2,
            concat=lambda x, y: np.concatenate([np.array(x), np.array(y)]),
            diff=lambda x, y: np.array(x) - np.array(y),
            abs_diff=lambda x, y: np.abs(np.array(x) - np.array(y)),
            mult=lambda x, y: np.array(x) * np.array(y),
        )
        return aggs[self.agg_name]

    @property
    @functools.lru_cache()
    def imdb_ids(self) -> Set[str]:
        path = self.path_data / Path(f"{self.split}.txt")
        return set(pd.read_fwf(path, header=None)[0])

    @staticmethod
    def _get_pairs_data_from_json(
        path: str, label_extractor: Callable[[dict], bool], source: Optional[str]
    ) -> List[Pair]:
        with open(path) as f:
            data = json.load(f)
        pairs = (
            Pair(
                imdb_id=x["imdb_id"],
                shot1_idx=x["clip1_index"],
                shot2_idx=x["clip2_index"],
                label=label_extractor(x),
                source_info=SourceInfo(
                    tuple(
                        (str(source["source_name"]), float(source["score"]))
                        for source in x["source_info"]
                    )
                )
                if "source_info" in x
                else None,
            )
            for x in data
        )
        return [p for p in pairs if source is None or p.source_info.from_source(source)]

    @property
    def pairs_labeled_all(self) -> List[Pair]:
        path = self.path_data / Path(f"dataset-{self.task}.json")
        return self._get_pairs_data_from_json(
            str(path), lambda x: x["majority_label"], source=self.source
        )

    @property
    def pairs_labeled(self) -> List[Pair]:
        return [p for p in self.pairs_labeled_all if p.imdb_id in self.imdb_ids]

    @property
    def pairs_random_negative_all(self) -> List[Pair]:
        path = self.path_data / Path("dataset-random-negatives.json")
        return self._get_pairs_data_from_json(str(path), lambda x: False, source=None)

    @property
    def pairs_random_negative(self) -> List[Pair]:
        return [p for p in self.pairs_random_negative_all if p.imdb_id in self.imdb_ids]

    @staticmethod
    def _shuffle(pairs: Sequence[Pair], random_state: Optional[int]) -> Tuple[Pair]:
        if random_state is not None:
            np.random.seed(random_state)
            pairs = np.random.choice(pairs, len(pairs), replace=False)
        return tuple(pairs)

    @property
    @functools.lru_cache()
    def pairs(self) -> Tuple[Pair]:
        pairs = (
            [p for p in self.pairs_labeled if p.label]
            if self.positive_only
            else self.pairs_labeled
        )
        return self._shuffle(pairs, random_state=self.random_state)

    @property
    @functools.lru_cache()
    def embs_expanded(self) -> FeatureMatrix:
        return np.vstack(
            [
                self.encoder(p.imdb_id, getattr(p, f"shot{shot_nbr}_idx"))
                for p in self.pairs
                for shot_nbr in (1, 2)
            ]
        )

    @property
    @functools.lru_cache()
    def x(self) -> FeatureMatrix:
        if self.encoder_name is None or self.agg_name is None:
            raise ValueError("cannot create a feature matrix without encoder and agg")
        return np.vstack(
            [
                self.agg(
                    self.encoder(p.imdb_id, p.shot1_idx),
                    self.encoder(p.imdb_id, p.shot2_idx),
                )
                for p in self.pairs
            ]
        )

    @property
    @functools.lru_cache()
    def y(self) -> Tuple[Label]:
        return tuple(p.label for p in self.pairs)

    @property
    @functools.lru_cache()
    def ids(self) -> Tuple[str]:
        return tuple(p.imdb_id for p in self.pairs)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def source_scores(self) -> Tuple[float]:
        return tuple(
            p.source_info.get_score(self.source) if p.source_info is not None else 0.0
            for p in self.pairs
        )


class MetricLearningDataset(TorchDataset):
    def __init__(
        self,
        ds: Dataset,
        ds_train: Dataset,
    ):
        self.ds = ds
        self.ds_train = ds_train
        self.indices = self._get_indices()
        embs_train = ds_train.embs_expanded
        self.mean = embs_train.mean(axis=0)
        self.std = embs_train.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        self.embs = ds.embs_expanded

    @classmethod
    def get_ds_train(cls, task: str, encoder: str) -> Dataset:
        return Dataset(
            split="train",
            task=task,
            encoder_name=encoder,
            random_state=None,
            positive_only=False,
        )

    @classmethod
    def by_args(
        cls, task: str, split: str, encoder: str, random_state: Optional[int] = None
    ) -> "MetricLearningDataset":
        ds_train_all = cls.get_ds_train(task, encoder)
        args = dict(
            split=split,
            task=task,
            encoder_name=encoder,
            random_state=random_state,
            positive_only=split == "train",
        )
        ds = Dataset(**args)
        return cls(ds, ds_train_all)

    def _get_indices(self) -> List[int]:
        return [i for idx in range(len(self.ds)) for i in [idx, idx]]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Embedding, int]:
        emb = (self.embs[idx] - self.mean) / self.std
        return emb.astype(np.float32), self.indices[idx]

    def dataloader(self, batch_size: int) -> DataLoader:
        # shuffling is done outside of the dataloader
        return DataLoader(self, batch_size=batch_size, shuffle=False)
