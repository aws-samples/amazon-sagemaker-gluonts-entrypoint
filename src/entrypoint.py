# Based on glounts/nursery/sagemaker_sdk/entrypoint_scripts/train_entry_point.py
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository import datasets
from gluonts.evaluation import backtest

from sm_util import MyEvaluator, hp2estimator, mkdir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

# TODO: implement model_fn, input_fn, predict_fn, and output_fn !!
# TODO: segment script for readability

# FIXME: logging stuffs: some function use logging.xxx() -> root logger.
# FIXME: at the begnning of script, check for logging handler, and add appropriately, and see if elapsed time etc.
#        appear when we python entrypoint.py ... 2>&1 | grep ...

# Training & batch transform has different logging handler.
if logging.root.handlers == []:
    # Training: no logging handler, so we need to setup one to stdout.
    # Reason to use stdout: xgboost script mode swallows stderr.
    print("Add logging handle to stdout")
    ch = logging.StreamHandler(sys.stdout)
    print("1000: created stream handler")
    ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    print("2000: formatted stream handler")
    logger.addHandler(ch)
    print("3000: added stream handler to logger")


def print_logging_setup(logger):
    """Walkthrough logger hierarchy and print details of each logger.

    Print to stdout to make sure CloudWatch pick it up, regardless of how logger handler is setup.
    """
    lgr = logging.getLogger(__name__)
    while lgr is not None:
        print("level: {}, name: {}, handlers: {}".format(lgr.level, lgr.name, lgr.handlers))
        lgr = lgr.parent


print_logging_setup(logger)


def train(args, algo_args):
    """Train a specified estimator on a specified dataset."""
    # Load data
    if args.s3_dataset is None:
        # load built in dataset
        logger.info("Downloading dataset %s", args.dataset)
        dataset = datasets.get_dataset(args.dataset)
    else:
        # load custom dataset
        logger.info("Loading dataset from %s", args.s3_dataset)
        s3_dataset_dir = Path(args.s3_dataset)
        dataset = load_datasets(metadata=s3_dataset_dir, train=s3_dataset_dir / "train", test=s3_dataset_dir / "test",)

    # Initialize estimator
    estimator = hp2estimator(args.algo, algo_args, dataset.metadata)
    logger.info("Estimator: %s", estimator)

    # Debug/dev/test milestone
    if args.stop_before == "train":
        logger.info("Early termination: before %s", args.stop_before)
        return

    # Train
    logger.info("Starting model training.")
    predictor = estimator.train(dataset.train)
    # Save
    model_dir = mkdir(args.model_dir)
    predictor.serialize(model_dir)

    # Debug/dev/test milestone
    if args.stop_before == "eval":
        logger.info("Early termination: before %s", args.stop_before)
        return

    # Backtesting
    logger.info("Starting model evaluation.")
    forecast_it, ts_it = backtest.make_evaluation_predictions(
        dataset=dataset.test, predictor=predictor, num_samples=args.num_samples,
    )

    # Compute standard metrics over all samples or quantiles, and plot each timeseries, all in one go!
    plot_dir = mkdir(Path(args.output_data_dir) / "plots")
    evaluator = MyEvaluator(
        plot_dir=plot_dir, ts_count=len(dataset.test), quantiles=args.quantiles, plot_transparent=args.plot_transparent
    )
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset.test))

    # required for metric tracking.
    for name, value in agg_metrics.items():
        logger.info(f"gluonts[metric-{name}]: {value}")

    # save the evaluation results
    metrics_output_dir = Path(args.output_data_dir)
    with open(metrics_output_dir / "agg_metrics.json", "w") as f:
        json.dump(agg_metrics, f)
    with open(metrics_output_dir / "item_metrics.csv", "w") as f:
        item_metrics.to_csv(f, index=False)


def parse_hyperparameters(hm) -> Dict[str, Any]:
    """Convert list of ['--name', 'value', ...] to { 'name': value}, where 'value' is converted to the nearest data type.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    """
    d = {}
    it = iter(hm)
    try:
        while True:
            key = next(it)[2:]
            value = next(it)
            d[key] = value
    except StopIteration:
        pass

    # Infer data types.
    dd = {k: infer_dtype(v) for k, v in d.items()}
    return dd


def infer_dtype(s):
    """Auto-cast string values to nearest matching datatype.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    Note that python 3.6 implements PEP-515 which allows '_' as thousand separators. Hence, on Python 3.6,
    '1_000' is a valid number and will be converted accordingly.
    """
    if s == "None":
        return None
    if s == "True":
        return True
    if s == "False":
        return False

    try:
        i = float(s)
        if ("." in s) or ("e" in s.lower()):
            return i
        else:
            return int(s)
    except:  # noqa:E722
        pass

    try:
        # If string is json, deser it.
        return json.loads(s)
    except:  # noqa:E722
        return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument("--sm-hps", type=json.loads, default=os.environ.get("SM_HPS", {}))

    # SageMaker protocols: input data, output dir and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    parser.add_argument("--s3-dataset", type=str, default=os.environ.get("SM_CHANNEL_S3_DATASET", None))
    parser.add_argument("--dataset", type=str, default=os.environ.get("SM_HP_DATASET", ""))

    # Arguments for evaluators
    parser.add_argument("--num_samples", type=int, default=os.environ.get("SM_HP_NUM_SAMPLES", 100))
    parser.add_argument(
        "--quantiles", default=os.environ.get("SM_HP_QUANTILES", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    )
    parser.add_argument(
        "--algo", type=str, default=os.environ.get("SM_HP_ALGO", "gluonts.model.deepar.DeepAREstimator")
    )

    # Argumets for plots
    parser.add_argument("--plot_transparent", type=int, default=os.environ.get("SM_HP_PLOT_TRANSPARENT", 0))

    # Debug/dev/test features; source code is the documentation hence, only for developers :).
    parser.add_argument("--stop_before", type=str, default="")

    logger.info("CLI args to entrypoint script: %s", sys.argv)
    args, train_args = parser.parse_known_args()
    algo_args = parse_hyperparameters(train_args)
    train(args, algo_args)
