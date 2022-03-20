import unittest

from sklearn.metrics import average_precision_score

from matchcut import experiments


class ExperimentsTest(unittest.TestCase):
    def test_metrics(self):
        metrics = experiments.Metrics(
            true=(True, False, True, False, False),
            probs=((0.1, 0.9, 0.95, 0.1, 0.8),),
            keys=("x", "x", "y", "y", "y"),
            round_to=2,
            metric_fn=average_precision_score,
        )
        assert metrics.metric_dist().mean == 0.7
        assert metrics.p_at_10[0].mean == 0.5
        assert metrics.p_at_25[0].mean == 0.5

    def test_heuristics(self):
        exp = experiments.Experiment1()
        assert exp.heuristic1().metrics_test.metrics_dict("test")
        assert exp.heuristic2().metrics_test.metrics_dict("test")
        assert exp.heuristic4().metrics_test.metrics_dict("test")
        assert exp.heuristic5().metrics_test.metrics_dict("test")

    def test_baseline(self):
        exp = experiments.Experiment1()
        assert exp.baseline("frame").metrics_test.metrics_dict("test")
        assert exp.baseline("motion").metrics_test.metrics_dict("test")
