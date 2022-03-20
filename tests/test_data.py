import unittest

from matchcut import data


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ds_dict = {
            split: data.Dataset(
                split="train",
                task="frame",
                encoder_name="clip4clip",
                agg_name="mean",
            )
            for split in ["train", "test", "validation"]
        }
        self.ds_motion = data.Dataset(
            split="train",
            task="motion",
            encoder_name="clip4clip",
            agg_name="concat",
        )

    def test_imdb_ids(self):
        assert len(self.ds_dict["train"].imdb_ids) == 60
        assert len(self.ds_dict["validation"].imdb_ids) == 60
        assert len(self.ds_dict["test"].imdb_ids) == 60
        assert "tt0385004" in self.ds_dict["train"].imdb_ids

    def test_paired_data(self):
        pairs = self.ds_dict["train"].pairs_labeled
        pairs_all = self.ds_dict["train"].pairs_labeled_all
        assert len(pairs) == 5993
        assert len(pairs_all) == 9985
        assert len(self.ds_motion.pairs_labeled_all) == 9320
