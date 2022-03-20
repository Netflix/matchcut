import functools
import itertools
from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import toolz

import pandas as pd
import numpy as np

from .config import common, experiment1, experiment2
from .data import Dataset, Label, RankingMetricFn
from .models import MetricLearning, Model


Probs = Tuple[float, ...]


@dataclass(frozen=True)
class Dist:
    """
    A collection of values that represent a distribution.
    """

    values: Tuple[Optional[float], ...]
    round_to: int

    @property
    def mean(self) -> float:
        return pd.Series(self.values).mean()

    @property
    def std(self) -> float:
        return pd.Series(self.values).std()

    @property
    def text(self) -> str:
        m = round(self.mean, self.round_to)
        return (
            str(m)
            if len(self.values) == 1
            else f"{m} ± {round(self.std, self.round_to)}"
        )


@dataclass(frozen=True)
class Metrics:
    true: Tuple[Label, ...]
    probs: Tuple[Probs, ...]
    keys: Tuple[str, ...]
    round_to: int
    metric_fn: RankingMetricFn

    def metric_dist(self, round_to: int = None) -> Dist:
        return Dist(
            values=tuple(self.metric_fn(self.true, ps) for ps in self.probs),
            round_to=round_to or self.round_to,
        )

    def metrics_dict(self, prefix: str, round_to: int = None) -> Dict[str, str]:
        return {f"{prefix}-AP": self.metric_dist(round_to=round_to).text}

    @property
    @functools.lru_cache()
    def metric_by_key(self) -> List[Dist]:
        return self._by_key(fn=self.metric_fn)

    @property
    @functools.lru_cache()
    def p_at_10(self) -> List[Dist]:
        fn = functools.partial(self._precision_at_k, k=10)
        return self._by_key(fn=fn)

    @property
    @functools.lru_cache()
    def p_at_25(self) -> List[Dist]:
        fn = functools.partial(self._precision_at_k, k=25)
        return self._by_key(fn=fn)

    @staticmethod
    def _precision_at_k(
        true: Sequence[Label], probs: Sequence[float], k: int
    ) -> Optional[float]:
        """
        returns NaN if no true positives exist.
        """
        k = min(sum(true), k)
        df = pd.DataFrame({"true": true, "probs": probs})
        return (
            df.sort_values(by="probs", ascending=False).head(k).true.mean()
            if k > 0
            else None
        )

    def _by_key(self, fn: RankingMetricFn) -> List[Dist]:
        return [
            Dist(
                values=tuple(
                    fn(rows.true, rows.probs)
                    for _, rows in pd.DataFrame(
                        dict(true=self.true, probs=probs, keys=self.keys)
                    ).groupby("keys")
                ),
                round_to=self.round_to,
            )
            for probs in self.probs
        ]


@dataclass(frozen=True)
class ResultRow:
    experiment: int
    task: str
    encoder: str
    model: Optional[str]
    agg: Optional[str]
    metrics_train: Metrics
    metrics_validation: Metrics
    metrics_test: Metrics

    def table_row(self, include_metrics: Tuple[str], round_to: int) -> dict:
        base = dict(
            task=self.task,
            encoder=self.encoder,
            model=self.model,
            agg=self.agg,
        )
        metrics = (
            getattr(self, f"metrics_{metric_name}").metrics_dict(
                prefix=metric_name, round_to=round_to
            )
            for metric_name in include_metrics
        )
        return {**base, **toolz.merge(*metrics)}


@dataclass(frozen=True)
class Experiment:
    experiment: int
    task: str
    encoder: str
    model: Optional[str]
    agg: Optional[str]
    random_state: int

    def exists(self) -> bool:
        return self.cache_path.exists()

    @property
    def cache_path(self) -> Path:
        path = Path(
            f"results/experiment{self.experiment}-{self.task}-{self.encoder}-{self.model}-{self.agg}-{self.random_state}.pkl"
        )
        path.parent.mkdir(exist_ok=True)
        return path

    @property
    def results(self) -> ResultRow:
        return pickle.load(open(self.cache_path, "rb"))

    def cache(self, res: ResultRow) -> None:
        return pickle.dump(res, open(self.cache_path, "wb"))


@dataclass(frozen=True)
class Experiment1:
    encoders: Tuple[str] = experiment1.encoders
    models: Tuple[str] = experiment1.models
    aggs: Tuple[str] = experiment1.aggs
    tasks: Tuple[str] = common.tasks
    random_state: int = common.random_state
    include_metrics: Tuple[str] = common.include_metrics
    cache: bool = common.cache
    round_to: int = common.round_to
    experiment: int = field(init=False, default=1)
    metric_fn: RankingMetricFn = common.metric_fn
    bootstrap_cnt: int = common.bootstrap_cnt
    quick: bool = False

    def _populate_results(
        self,
        task: str,
        encoder: str,
        model: Optional[str],
        agg: Optional[str],
        dataset_dict: Dict[str, Dataset],
        probs_train: Sequence[Sequence[float]],
        probs_validation: Sequence[Sequence[float]],
        probs_test: Sequence[Sequence[float]],
    ) -> ResultRow:
        metric = functools.partial(
            Metrics, round_to=self.round_to, metric_fn=self.metric_fn
        )
        return ResultRow(
            experiment=self.experiment,
            task=task,
            encoder=encoder,
            model=model,
            agg=agg,
            metrics_train=metric(
                dataset_dict["train"].y, probs_train, dataset_dict["train"].ids
            ),
            metrics_validation=metric(
                dataset_dict["validation"].y,
                probs_validation,
                dataset_dict["validation"].ids,
            ),
            metrics_test=metric(
                dataset_dict["test"].y, probs_test, dataset_dict["test"].ids
            ),
        )

    @staticmethod
    def _get_dataset_dict(
        task: str,
        encoder_name: Optional[str],
        agg_name: Optional[str],
        source: Optional[str] = None,
    ) -> Dict[str, Dataset]:
        return {
            split: Dataset(split, task, encoder_name, agg_name, source)
            for split in ("train", "validation", "test")
        }

    @staticmethod
    def _bootstrap(
        x: np.ndarray, y: Sequence[Label], seed: int
    ) -> Tuple[np.ndarray, List[Label]]:
        np.random.seed(seed)
        n = len(x)
        idx = np.random.choice(range(n), n, replace=True)
        return x[idx], list(np.array(y)[idx])

    def _run_experiment(
        self,
        task: str,
        encoder_name: str,
        model_name: str,
        agg_name: str,
        bootstrap_cnt: int,
        metric_fn: Callable,
    ) -> ResultRow:
        exp = Experiment(
            self.experiment, task, encoder_name, model_name, agg_name, self.random_state
        )
        if self.cache and exp.exists():
            return exp.results
        print(
            f"Running experiment task={task}, encoder={encoder_name}, model={model_name}, agg={agg_name}"
        )
        dataset_dict = self._get_dataset_dict(task, encoder_name, agg_name)

        preds_train = []
        preds_validation = []
        models = []
        scores = []

        for bootstrap_idx in range(bootstrap_cnt):
            model = Model(model_name=model_name, random_state=self.random_state)
            x, y = self._bootstrap(
                dataset_dict["train"].x, dataset_dict["train"].y, seed=bootstrap_idx
            )
            model.fit(x, y)
            models.append(model)
            preds_train.append(model.predict_proba(dataset_dict["train"].x))
            preds_validation.append(model.predict_proba(dataset_dict["validation"].x))
            scores.append(metric_fn(dataset_dict["validation"].y, preds_validation[-1]))

        preds_test = tuple(
            models[np.argmax(scores)].predict_proba(dataset_dict["test"].x)
        )

        res = self._populate_results(
            task,
            encoder_name,
            model_name,
            agg_name,
            dataset_dict,
            tuple(preds_train),
            tuple(preds_validation),
            (preds_test,),
        )
        if self.cache:
            exp.cache(res)
        return res

    def run(self) -> pd.DataFrame:
        res = [
            self.baseline(task="frame"),
            self.baseline(task="motion"),
            self.heuristic1(),
            self.heuristic2(),
            self.heuristic3(),
            self.heuristic4(),
            self.heuristic5(),
        ]
        params = itertools.product(self.tasks, self.encoders, self.models, self.aggs)
        params_to_run = itertools.islice(params, 1 if self.quick else None)
        experiments = (
            self._run_experiment(
                task=task,
                encoder_name=encoder,
                model_name=model,
                agg_name=agg,
                bootstrap_cnt=self.bootstrap_cnt,
                metric_fn=self.metric_fn,
            )
            for task, encoder, model, agg in params_to_run
        )
        res = [
            exp.table_row(include_metrics=self.include_metrics, round_to=self.round_to)
            for exp in experiments
        ]
        return pd.DataFrame(res)

    def baseline(self, task: str) -> ResultRow:
        print(f"Running baseline experiment task={task}")
        fn = np.random.rand
        np.random.seed(self.random_state)
        dataset_dict = self._get_dataset_dict(task, None, None)
        probs_train = tuple(fn(len(dataset_dict["train"])))
        probs_validation = tuple(fn(len(dataset_dict["validation"])))
        probs_test = tuple(fn(len(dataset_dict["test"])))
        return self._populate_results(
            task,
            "baseline",
            None,
            None,
            dataset_dict,
            (probs_train,),
            (probs_validation,),
            (probs_test,),
        )

    def _heuristic(self, task: str, source: str) -> ResultRow:
        print(f"Running heuristic experiment task={task}, heuristic={source}")
        dataset_dict = self._get_dataset_dict(task, None, None, source=source)
        return self._populate_results(
            task,
            source,
            None,
            None,
            dataset_dict,
            (dataset_dict["train"].source_scores,),
            (dataset_dict["validation"].source_scores,),
            (dataset_dict["test"].source_scores,),
        )

    def heuristic1(self) -> ResultRow:
        return self._heuristic("frame", "heuristic 1")

    def heuristic2(self) -> ResultRow:
        return self._heuristic("frame", "heuristic 2")

    def heuristic3(self) -> ResultRow:
        task = "frame"
        print(f"Running heuristic experiment task={task}, heuristic=heuristic 3")
        dataset_dict = self._get_dataset_dict(task, None, None, None)
        df_scores = pd.read_csv("data/heuristic3-scores.csv")
        scores = {
            (row.imdb_id, row.shot1_idx, row.shot2_idx): row.iiou_score
            for _, row in df_scores.iterrows()
        }
        scores_dict = {
            split: tuple(
                scores[(pair.imdb_id, pair.shot1_idx, pair.shot2_idx)]
                for pair in dataset_dict[split].pairs
            )
            for split, ds in dataset_dict.items()
        }
        return self._populate_results(
            task,
            "heuristic 3",
            None,
            None,
            dataset_dict,
            (scores_dict["train"],),
            (scores_dict["validation"],),
            (scores_dict["test"],),
        )

    def heuristic4(self) -> ResultRow:
        return self._heuristic("motion", "heuristic 4")

    def heuristic5(self) -> ResultRow:
        return self._heuristic("motion", "heuristic 5")


@dataclass(frozen=True)
class Experiment2:
    task_encoders: Tuple[Tuple[str, str]] = experiment2.task_encoders
    hyperparameters: experiment2.TaskParameters = experiment2.hyperparameters
    type_of_triplets: str = experiment2.type_of_triplets
    batch_size_train: int = experiment2.batch_size_train
    batch_size_evaluation: int = experiment2.batch_size_evaluation
    seed_step: int = experiment2.seed_step
    include_metrics: Tuple[str] = common.include_metrics
    random_state: int = common.random_state
    cache: bool = common.cache
    round_to: int = common.round_to
    bootstrap_cnt: int = common.bootstrap_cnt
    experiment: int = field(init=False, default=2)
    metric_fn: RankingMetricFn = common.metric_fn
    quick: bool = False

    @staticmethod
    def combine_metrics(metrics: Sequence[Metrics]) -> Metrics:
        first = metrics[0]
        return Metrics(
            true=first.true,
            probs=tuple(m.probs[0] for m in metrics),
            keys=first.keys,
            round_to=first.round_to,
            metric_fn=first.metric_fn,
        )

    def _run_experiment(self, task: str, encoder: str) -> ResultRow:
        exp = Experiment(self.experiment, task, encoder, None, None, self.random_state)
        if self.cache and exp.exists():
            return exp.results
        print(f"Running experiment task={task}, encoder={encoder}")
        hps = getattr(self.hyperparameters, task)
        results: List[ResultRow] = []
        for idx in range(self.bootstrap_cnt):
            ml = MetricLearning(
                task,
                encoder,
                hps.temperature,
                self.type_of_triplets,
                common.embedding_dims[encoder],
                hps.mlp_dim1,
                hps.mlp_dim2,
                hps.mlp_lr,
                hps.weight_decay,
                hps.epochs,
                self.batch_size_train,
                self.batch_size_evaluation,
                self.random_state + idx,
                self.seed_step,
            )
            ml.train()
            res = ResultRow(
                self.experiment,
                task,
                encoder,
                None,
                None,
                metrics_train=self._get_metrics(ml, "train"),
                metrics_validation=self._get_metrics(ml, "validation"),
                metrics_test=self._get_metrics(ml, "test"),
            )
            results.append(res)

        # aggregate metrics
        first_res = results[0]
        metrics_validation = self.combine_metrics(
            [r.metrics_validation for r in results]
        )
        scores_validation = metrics_validation.metric_dist(
            round_to=self.round_to
        ).values
        res = ResultRow(
            experiment=first_res.experiment,
            task=first_res.task,
            encoder=first_res.encoder,
            model=first_res.model,
            agg=first_res.agg,
            metrics_train=self.combine_metrics([r.metrics_train for r in results]),
            metrics_validation=metrics_validation,
            metrics_test=results[np.argmax(scores_validation)].metrics_test,
        )

        if self.cache:
            exp.cache(res)
        return res

    def _get_metrics(self, ml: MetricLearning, split: str) -> Metrics:
        ds = getattr(ml, f"dataset_{split}")
        return Metrics(
            true=ds.y,
            probs=(getattr(ml, f"probs_{split}"),),
            keys=ds.ids,
            round_to=self.round_to,
            metric_fn=self.metric_fn,
        )

    def run(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._run_experiment(task, encoder).table_row(
                include_metrics=self.include_metrics,
                round_to=self.round_to,
            )
            for idx, (task, encoder) in enumerate(self.task_encoders)
            if not self.quick or idx == 0
        )
