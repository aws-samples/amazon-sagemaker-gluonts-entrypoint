# Based on glounts/nursery/sagemaker_sdk/entrypoint_scripts/train_entry_point.py
# TODO: implement model_fn, input_fn, predict_fn, and output_fn !!
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.cbook
import numpy as np
from gluonts.dataset.common import DataEntry, ListDataset, TrainDatasets
from gluonts.model.forecast import Config, Forecast
from gluonts.model.predictor import Predictor

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# try block to prevent isort shifting the import statements.
# See also: https://github.com/timothycrosley/isort/issues/295#issuecomment-570898035
try:
    # region: quiet tqdm

    # This stanza must appear before any module that uses tqdm.
    # https://github.com/tqdm/tqdm/issues/619#issuecomment-425234504

    # By default, disable tqdm if we believe we're running under SageMaker. This has to be done before importing any
    # other module that uses tqdm.
    if "SM_HOSTS" in os.environ:
        print("Env. var SM_HOSTS detected. Silencing tqdm as we're likely to run on SageMaker...")
        import tqdm
        from tqdm import auto as tqdm_auto

        old_auto_tqdm = tqdm_auto.tqdm

        def nop_tqdm_off(*a, **k):
            k["disable"] = True
            return old_auto_tqdm(*a, **k)

        tqdm_auto.tqdm = (
            nop_tqdm_off  # For download, completely disable progress bars: large models, lots of stuffs printed.
        )

        # Used by run_ner.py
        old_tqdm = tqdm.tqdm

        def nop_tqdm(*a, **k):
            k["ncols"] = 0
            k["mininterval"] = 3600
            return old_tqdm(*a, **k)

        tqdm.tqdm = nop_tqdm

        # Used by run_ner.py
        old_trange = tqdm.trange

        def nop_trange(*a, **k):
            k["ncols"] = 0
            k["mininterval"] = 3600
            return old_trange(*a, **k)

        tqdm.trange = nop_trange
    # endregion: quiet tqdm

    from gluonts.dataset.common import load_datasets
    from gluonts.dataset.repository import datasets
    from gluonts.evaluation import backtest

    from sm_util import hp2estimator, mkdir
    from evaluator import MyEvaluator
except:  # noqa: E722
    raise

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)

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
# logger.setLevel(logging.DEBUG)


def log1p_tds(dataset: TrainDatasets) -> TrainDatasets:
    """Create a new train datasets with targets log-transformed."""
    # Implementation note: currently, the only way is to eagerly load all timeseries in memory, and do the transform.
    train = ListDataset(dataset.train, freq=dataset.metadata.freq)
    log1p(train)

    if dataset.test is not None:
        test = ListDataset(dataset.test, freq=dataset.metadata.freq)
        log1p(test)
    else:
        test = None

    # fmt: off
    return TrainDatasets(
        dataset.metadata.copy(),  # Note: pydantic's deep copy.
        train=train,
        test=test
    )
    # fmt: on


def log1p(ds: ListDataset):
    """In-place log transformation."""
    for data_entry in ds:
        data_entry["target"] = np.log1p(data_entry["target"])


def expm1_and_clip_to_zero(_, yhat: np.ndarray):
    """Expm1, followed by clip at 0.0."""
    logger.debug("Before expm1: %s %s", yhat.shape, yhat)
    logger.debug("After expm1: %s %s", yhat.shape, np.expm1(yhat))

    return np.clip(np.expm1(yhat), a_min=0.0, a_max=None)


def clip_to_zero(_, yhat: np.ndarray):
    return np.clip(yhat, a_min=0.0, a_max=None)


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

    # Initialize estimator
    estimator = hp2estimator(args.algo, algo_args, dataset.metadata)
    logger.info("Estimator: %s", estimator)

    # Debug/dev/test milestone
    if args.stop_before == "train":
        logger.info("Early termination: before %s", args.stop_before)
        return

    # Train
    logger.info("Starting model training.")
    predictor = estimator.train(training_data=dataset.train, validation_data=dataset.test)
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


def model_fn(model_dir: Union[str, Path]) -> Predictor:
    """Load a glounts model from a directory.

    Args:
        model_dir (Union[str, Path]): a directory where model is saved.

    Returns:
        Predictor: A gluonts predictor.
    """
    predictor = Predictor.deserialize(Path(model_dir))

    # If model was trained on log-space, then forecast must be inverted before metrics etc.
    with open(os.path.join(model_dir, "y_transform.json"), "r") as f:
        y_transform = json.load(f)
        logger.info("model_fn: custom transformations = %s", y_transform)

        if y_transform["inverse_transform"] == "expm1":
            predictor.output_transform = expm1_and_clip_to_zero
        else:
            predictor.output_transform = clip_to_zero

        # Custom field
        predictor.pre_input_transform = log1p if y_transform["transform"] == "log1p" else None

    logger.info("predictor.pre_input_transform: %s", predictor.pre_input_transform)
    logger.info("predictor.output_transform: %s", predictor.output_transform)
    logger.info("model_fn() done; loaded predictor %s", predictor)

    return predictor


def input_fn(request_body: str, request_content_type: str = "") -> List[DataEntry]:
    """Deserialize JSON-lines into Python objects.

    Args:
        request_body (str): Incoming payload.
        request_content_type (str, optional): Ignored. Defaults to "".

    Returns:
        List[DataEntry]: List of gluonts timeseries.
    """
    import io
    import json

    return [json.loads(line) for line in io.StringIO(request_body)]


def predict_fn(input_object: List[DataEntry], model: Predictor, num_samples=1000) -> List[Forecast]:
    """Take the deserialized JSON-lines, then perform inference against the loaded model.

    Args:
        input_object (List[DataEntry]): List of gluonts timeseries.
        model (Predictor): A gluonts predictor.
        num_samples (int, optional): Number of forecast paths for each timeseries. Defaults to 1000.

    Returns:
        List[Forecast]: List of forecast results.
    """
    from gluonts.dataset.common import ListDataset

    # Create ListDataset here, because we need to match their freq with model's freq.
    X = ListDataset(input_object, freq=model.freq)

    # Apply forward transformation to input data, before injecting it to the predictor.
    if model.pre_input_transform is not None:
        logger.debug("Before model.pre_input_transform: %s", X.list_data)
        model.pre_input_transform(X)
        logger.debug("After model.pre_input_transform: %s", X.list_data)

    it = model.predict(X, num_samples=num_samples)
    return list(it)


def output_fn(
    forecasts: List[Forecast],
    content_type: str = "",
    config: Config = Config(quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
) -> str:
    """Take the prediction result and serializes it according to the response content type.

    Args:
        prediction (List[Forecast]): List of forecast results.
        content_type (str, optional): Ignored. Defaults to "".

    Returns:
        List[str]: List of JSON-lines, each denotes forecast results in quantiles.
    """
    return "\n".join((json.dumps(forecast.as_json_dict(config)) for forecast in forecasts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # an alternative way to load hyperparameters via SM_HPS environment variable.
    parser.add_argument("--sm-hps", type=json.loads, default=os.environ.get("SM_HPS", {}))

    # SageMaker protocols: input data, output dir and model directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    parser.add_argument("--s3_dataset", type=str, default=os.environ.get("SM_CHANNEL_S3_DATASET", None))
    parser.add_argument("--dataset", type=str, default=os.environ.get("SM_HP_DATASET", ""))

    # Arguments for evaluators
    parser.add_argument("--num_samples", type=int, default=os.environ.get("SM_HP_NUM_SAMPLES", 1000))
    parser.add_argument(
        "--quantiles", default=os.environ.get("SM_HP_QUANTILES", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    )
    parser.add_argument(
        "--algo", type=str, default=os.environ.get("SM_HP_ALGO", "gluonts.model.deepar.DeepAREstimator")
    )
    parser.add_argument("--y_transform", type=str, default="noop", choices=["noop", "log1p"])

    # Argumets for plots
    parser.add_argument("--plot_transparent", type=int, default=os.environ.get("SM_HP_PLOT_TRANSPARENT", 0))

    # Debug/dev/test features; source code is the documentation hence, only for developers :).
    parser.add_argument("--stop_before", type=str, default="")

    logger.info("CLI args to entrypoint script: %s", sys.argv)
    args, train_args = parser.parse_known_args()
    algo_args = parse_hyperparameters(train_args)
    train(args, algo_args)
