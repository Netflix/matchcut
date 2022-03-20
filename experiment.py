import argparse

from matchcut import experiments
from matchcut.config import experiment1, experiment2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--number", choices=[1, 2], type=int, help="Pass experiment 1 or 2"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    exp1 = int(args.number) == 1
    cls = experiments.Experiment1 if exp1 else experiments.Experiment2
    quick = args.quick
    df = cls(quick=quick).run()
    path_out = getattr(experiment1 if exp1 else experiment2, "path_output")
    df.to_csv(path_out, index=False)
