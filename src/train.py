import smepu

import inspect
import json
import os
import sys
import warnings
from pathlib import Path
from pydoc import locate
from typing import Any, Dict

import matplotlib.cbook
import numpy as np
from gluonts.dataset.common import load_datasets
from gluonts.dataset.repository import datasets
from gluonts.evaluation import backtest
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from gluonts_example.evaluator import MyEvaluator
from gluonts_example.util import clip_to_zero, expm1_and_clip_to_zero, log1p_tds, mkdir

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# Setup logger must be done in the entrypoint script.
logger = smepu.setup_opinionated_logger(__name__)


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
        dataset = load_datasets(
            metadata=s3_dataset_dir / "metadata", train=s3_dataset_dir / "train", test=s3_dataset_dir / "test",
        )
        # Apply transformation if requested
        if args.y_transform == "log1p":
            dataset = log1p_tds(dataset)

    # Estimator is an instance of "algo" class.
    klass: Any = locate(args.algo)
    estimator = klass(**algo_args)
    logger.info("Estimator: %s", estimator)
    # Initialize estimator
    # estimator = hp2estimator(args.algo, algo_args, dataset.metadata)

    # Debug/dev/test milestone
    if args.stop_before == "train":
        logger.info("Early termination: before %s", args.stop_before)
        return

    # Probe the right kwarg for validation data.
    # - NPTSEstimator (or any based on DummyEstimator) uses validation_dataset=...
    # - Other estimators use validation_data=...
    candidate_kwarg = [k for k in inspect.signature(estimator.train).parameters if "validation_data" in k]
    kwargs = {"training_data": dataset.train}
    if len(candidate_kwarg) == 1:
        kwargs[candidate_kwarg[0]] = dataset.test
    else:
        kwargs["validation_data"] = dataset.test

    # Train
    logger.info("Starting model training.")
    # predictor = estimator.train(training_data=dataset.train, validation_data=dataset.test)
    predictor = estimator.train(**kwargs)
    # Save
    model_dir = mkdir(args.model_dir)
    predictor.serialize(model_dir)
    # Also record the y's transformation & inverse transformation.
    with open(os.path.join(args.model_dir, "y_transform.json"), "w") as f:
        if args.y_transform == "log1p":
            f.write('{"transform": "log1p", "inverse_transform": "expm1"}\n')
            predictor.output_transform = expm1_and_clip_to_zero
        else:
            f.write('{"transform": "noop", "inverse_transform": "clip_at_zero"}\n')
            predictor.output_transform = clip_to_zero

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


def freq_name(s):
    """Convert frequency string to friendly name.

    This implementation uses only frequency string, hence 7D still becomes daily. It's not smart enough yet to know
    that 7D equals to week.
    """
    offset = to_offset(s)
    if isinstance(offset, offsets.Day):
        return "daily"
    elif isinstance(offset, offsets.Week):
        return "weekly"
    raise ValueError(f"Unsupported frequency: {s}")


if __name__ == "__main__":
    # Minimal argparser for SageMaker protocols
    parser = smepu.argparse.sm_protocol(channels=["s3_dataset"])

    # Hyperparameters captured by this entrypoint script.
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

    logger.info("CLI args to entrypoint script: %s", sys.argv)
    args, train_args = parser.parse_known_args()

    # Convert cli args / hyperparameters to kwargs
    kwargs: Dict[str, Any] = smepu.argparse.to_kwargs(train_args)

    # algo_args = parse_hyperparameters(train_args)
    train(args, kwargs)
