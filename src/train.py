import smepu

import inspect
import json
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pydoc import locate
from typing import Any, Dict

import matplotlib.cbook
import numpy as np
from gluonts.dataset.common import TrainDatasets, load_datasets
from gluonts.dataset.repository import datasets
from gluonts.evaluation import backtest

from gluonts_example.evaluator import MyEvaluator
from gluonts_example.util import clip_to_zero, expm1_and_clip_to_zero, freq_name, log1p_tds, mkdir, override_hp

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)

INVERSE = {
    "log1p": expm1_and_clip_to_zero,
    "noop": clip_to_zero,
}


def train(args: Namespace, algo_args: Dict[str, Any]) -> None:
    """Train a specified estimator on a specified dataset."""
    dataset = load_dataset(args)
    algo_args = override_hp(algo_args, dataset.metadata)
    estimator = new_estimator(args.algo, kwargs=algo_args)

    # Debug/dev/test milestone
    if args.stop_before == "train":
        logger.info("Early termination: before %s", args.stop_before)
        return

    # Train & save model
    logger.info("Starting model training.")
    if args.y_transform == "log1p":
        dataset = log1p_tds(dataset)
    train_kwargs = get_train_kwargs(estimator, dataset)
    predictor = estimator.train(**train_kwargs)
    predictor.output_transform = INVERSE[args.y_transform]
    save_model(predictor, args)

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
    # Remember to specify gt_inverse_transform when computing metrics.
    logger.info("MyEvaluator: assume non-negative ground truths, hence no clip_to_zero performed on them.")
    gt_inverse_transform = np.expm1 if args.y_transform == "log1p" else None
    evaluator = MyEvaluator(
        out_dir=Path(args.output_data_dir),
        quantiles=args.quantiles,
        plot_transparent=bool(args.plot_transparent),
        gt_inverse_transform=gt_inverse_transform,
        clip_at_zero=True,
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

    # Specific requirement: output wmape to a separate file.
    with open(metrics_output_dir / f"{freq_name(dataset.metadata.freq)}-wmapes.csv", "w") as f:
        warnings.warn(
            "wmape csv uses daily or weekly according to frequency string, "
            "hence 7D still results in daily rather than weekly."
        )
        wmape_metrics = item_metrics[["item_id", "wMAPE"]].rename(
            {"item_id": "category", "wMAPE": "test_wMAPE"}, axis=1
        )
        wmape_metrics.to_csv(f, index=False)


def load_dataset(args: Namespace) -> TrainDatasets:
    """Load data from channel or fallback to named public dataset."""
    if args.s3_dataset is None:
        # load built in dataset
        logger.info("Downloading dataset %s", args.dataset)
        dataset = datasets.get_dataset(args.dataset)
    else:
        # load custom dataset
        logger.info("Loading dataset from %s", args.s3_dataset)
        s3_dataset_dir = Path(args.s3_dataset)
        dataset = load_datasets(
            metadata=s3_dataset_dir / "metadata", train=s3_dataset_dir / "train", test=s3_dataset_dir / "test",
        )
    return dataset


def new_estimator(algo: str, kwargs) -> Any:
    """Initialize an instance of klass with the specified kwargs."""
    klass: Any = locate(algo)
    estimator = klass(**kwargs)
    logger.info("Estimator: %s", estimator)
    return estimator


def get_train_kwargs(estimator, dataset) -> Dict[str, Any]:
    """Probe the right validation-data kwarg for the estimator.

    Known cases :

    - NPTSEstimator (or any other based on DummyEstimator) uses validation_dataset=...
    - Other estimators use validation_data=...
    """
    candidate_kwarg = [k for k in inspect.signature(estimator.train).parameters if "validation_data" in k]
    kwargs = {"training_data": dataset.train}
    if len(candidate_kwarg) == 1:
        kwargs[candidate_kwarg[0]] = dataset.test
    else:
        kwargs["validation_data"] = dataset.test
    return kwargs


def save_model(predictor, args: Namespace):
    predictor.serialize(mkdir(args.model_dir))
    with open(os.path.join(args.model_dir, "y_transform.json"), "w") as f:
        inverse = INVERSE[args.y_transform]
        f.write('{"transform": "%s", "inverse_transform": "%s"}\n' % (args.y_transform, inverse.__name__))


def add_args(parser: ArgumentParser):
    """Configure hyperparameters captured by this entrypoint script."""
    parser.add_argument(
        "--algo",
        type=str,
        help="Estimator class",
        default=os.environ.get("SM_HP_ALGO", "gluonts.model.deepar.DeepAREstimator"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="When s3_dataset channel not specified, fallback to this public dataset.",
        default=os.environ.get("SM_HP_DATASET", ""),
    )
    parser.add_argument(
        "--y_transform",
        type=str,
        help="Transformation to apply on target variable.",
        default="noop",
        choices=["noop", "log1p"],
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples for backtesting.",
        default=os.environ.get("SM_HP_NUM_SAMPLES", 1000),
    )
    parser.add_argument(
        "--quantiles",
        help="Quantiles for backtesting",
        default=os.environ.get("SM_HP_QUANTILES", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    )
    parser.add_argument(
        "--plot_transparent",
        type=int,
        help="Whether plots use transparent background.",
        default=os.environ.get("SM_HP_PLOT_TRANSPARENT", 0),
    )
    parser.add_argument("--stop_before", type=str, help="For debug/dev/test", default="", choices=["", "train", "eval"])


if __name__ == "__main__":
    # Minimal argparser for SageMaker protocols
    parser = smepu.argparse.sm_protocol(channels=["s3_dataset"])
    add_args(parser)

    logger.info("CLI args to entrypoint script: %s", sys.argv)
    args, train_args = parser.parse_known_args()

    train(args, smepu.argparse.to_kwargs(train_args))
